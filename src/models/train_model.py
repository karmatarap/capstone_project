import argparse
import logging
import math
import os
from collections import Counter
from typing import Iterable, Iterator, NamedTuple, Tuple

import cv2
import librosa

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset

from utils.common import load_dz_data
from utils.metrics import Metrics
from utils.test_split import TestSplitter

from .config import (
    AudioParams,
    best_params,
    data_params,
    spec_aug_combos,
    wav_aug_combos,
)
from .dataset import ElephantDataset
from .engine import Engine
from .models import get_pretrained_model
from .utils import get_fold, set_seeds, log_neptune


def run_train(
    fold: int,
    data_params: dict,
    hyper_params: dict,
    wav_aug_combos: dict,
    spec_aug_combos: dict,
    save_model=False,
    neptune_logger=None,
) -> float:

    logging.info(f"FOLD {fold} ---------------------")

    target_col = data_params["STRATIFY_COL"]
    epochs = hyper_params["epochs"]
    # Load Data
    df = load_dz_data(data_params["BASE_DATA_DIR"], target_col=target_col)
    n_classes = len(set(df[target_col]))
    # Create wav paths
    df["wav_path"] = "./data/raw/" + df["unique_ID"] + ".wav"
    lbl_enc = preprocessing.LabelEncoder()

    # Split data
    df_train, df_valid = get_fold(df, fold, output_path=data_params["OUTPUT_PATH"])

    # Encode target
    train_targets = lbl_enc.fit_transform(df_train[target_col])

    wav_augs = wav_aug_combos[hyper_params["wav_augs"]]
    spec_augs = spec_aug_combos[hyper_params["spec_augs"]]
    params = AudioParams()

    train_dataset = ElephantDataset(
        df_train.wav_path,
        train_targets,
        params,
        wav_augmentations=wav_augs,
        spec_augmentations=spec_augs,
    )
    train_dl = torch.utils.data.DataLoader(
        train_dataset, batch_size=12, num_workers=4  # , sampler=train_sampler
    )

    valid_targets = lbl_enc.fit_transform(df_valid[target_col])
    valid_dataset = ElephantDataset(
        df_valid.wav_path,
        valid_targets,
        params,
        wav_augmentations=wav_augs,
        spec_augmentations=spec_augs,
    )
    valid_dl = torch.utils.data.DataLoader(
        valid_dataset, batch_size=12, shuffle=False, num_workers=4
    )

    myModel = get_pretrained_model(hyper_params, num_classes=n_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    myModel = myModel.to(device)

    # Create class weights to combat imbalance
    class_sample_count = list(Counter(train_targets).values())

    norm_weights = [1 - (x / sum(class_sample_count)) for x in class_sample_count]
    norm_weights = torch.FloatTensor(norm_weights).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=norm_weights)

    learning_rate = hyper_params["learning_rate"]
    optimizer = torch.optim.Adam(myModel.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        steps_per_epoch=int(len(train_dl)),
        epochs=epochs,
        anneal_strategy="linear",
    )

    # Repeat for each epoch
    min_valid_loss = math.inf
    max_f1_macro = 0

    # Poor mans early stopping, abort training if validation loss does not
    # improve for x successive epochs where x is 10% of total epochs
    early_stopping_iter = epochs // 10
    early_stopping_counter = 0

    set_seeds(data_params["SEED"])

    engine = Engine(myModel, optimizer, scheduler, loss_fn, device)

    for epoch in range(epochs):
        model_path = f"{fold}-best-model-parameters.pt"
        train_loss = engine.train_one_epoch(train_dl)
        valid_loss, targets, predictions = engine.validate_one_epoch(valid_dl)
        m = Metrics(targets, predictions).score(average_="macro")
        # if epochs % 10 == 0:

        if neptune_logger is not None:
            neptune_logger[f"train/{fold}/loss"].log(train_loss)
            neptune_logger[f"valid/{fold}/loss"].log(valid_loss)

            for k, v in m.items():
                neptune_logger[f"valid/{fold}/{k}"].log(v)

        logging.info(
            f"Fold {fold} ,Training Loss: {train_loss}, Validation Loss: {valid_loss}, F1 Score: {m['F1_macro']}"
        )
        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            max_f1_macro = m["F1_macro"]
            if save_model:
                logging.info("Saving model.....")
                torch.save(myModel.state_dict(), model_path)
                neptune_logger[f"best_model_params_{fold}"].track_files(model_path)
        else:
            early_stopping_counter += 1
        if early_stopping_counter > early_stopping_iter:
            logging.info(f"Early stopping after {early_stopping_counter} iterations")
            break

    return min_valid_loss, max_f1_macro


def train_one_model(
    data_params=data_params,
    best_params=best_params,
    wav_params=wav_aug_combos,
    spec_params=spec_aug_combos,
    save_model=True,
    neptune_logger=None,
):
    all_losses, all_f1 = [], []
    for f in range(data_params["NUM_K_FOLDS"]):
        temp_loss, temp_f1 = run_train(
            f,
            data_params,
            best_params,
            wav_params,
            spec_params,
            save_model=save_model,
            neptune_logger=neptune_logger,
        )
        all_losses.append(temp_loss)
        all_f1.append(temp_f1)
    mean_loss = np.mean(all_losses)
    mean_f1 = np.mean(all_f1)
    logging.info(f"Mean Min Loss: {mean_loss}, Mean Max F1 macro {mean_f1}")
    return mean_loss, mean_f1


def objective(trial):
    # Objective function for optuna to minimize

    hyper_params = {
        "pretrained_model": trial.suggest_categorical(
            "pretrained_model", ["resnext50_32x4d", "resnet50", "efficientnet-b4"]
        ),
        "wav_augs": trial.suggest_categorical("wav_augs", list(wav_aug_combos.keys())),
        "spec_augs": trial.suggest_categorical(
            "spec_augs", list(spec_aug_combos.keys())
        ),
        "num_layers": trial.suggest_int(
            "num_layer", 0, 7
        ),  # additional layers after pretrained models
        "hidden_size": trial.suggest_int("hidden_size", 16, 2048),
        "dropout": trial.suggest_uniform("dropout", 0.1, 0.7),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-2),
        "seed": data_params["SEED"],
        "target": data_params["STRATIFY_COL"],
        "epochs": 50,
        "batch_size": 12,
    }

    with log_neptune() as run:
        run["parameters"] = hyper_params
        run["mode"] = "search"
        mean_loss, mean_f1 = train_one_model(
            best_params=hyper_params, save_model=False, neptune_logger=run
        )

        run["eval/mean_loss"] = mean_loss
        run["eval/mean_f1"] = mean_f1
    return mean_loss


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["search", "train"],
        default="train",
        help="Specify if hyperparameter search should be run, or best_param model should be trained.",
    )

    args = parser.parse_args()

    if args.mode == "search":
        logging.debug("Hyperparameter search mode")
        # Objective function is to minimize minimum validation loss
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=10)

        # Print and save parameters of best model
        trial_ = study.best_trial
        logging.info(f"Best trial values: {trial_.values}")
        logging.info(f"Best trial values: {trial_.params}")
        p = trail_.params
    else:
        logging.debug("Training best model from best_params in .config.py")
        logging.debug(f"Using parameters: {best_params}")
        p = best_params

    with log_neptune() as run:
        run["parameters"] = p
        run["mode"] = "train"
        mean_loss, mean_f1 = train_one_model(
            best_params=p, save_model=True, neptune_logger=run
        )

        run["eval/mean_loss"] = mean_loss
        run["eval/mean_f1"] = mean_f1
