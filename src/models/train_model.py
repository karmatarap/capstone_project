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
from utils.cross_validation import CrossValidator

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
from .utils import split_train_val_test, log_neptune, set_seeds


def run_train(
    data_params: dict,
    hyper_params: dict,
    wav_aug_combos: dict,
    spec_aug_combos: dict,
    save_model=False,
    neptune_logger=None,
) -> float:

    seed = data_params["SEED"]
    logging.info(f"SEED {seed} ---------------------")
    set_seeds(seed)
    target_col = data_params["STRATIFY_COL"]
    epochs = hyper_params["epochs"]
    # Load Data
    df = load_dz_data(data_params["BASE_DATA_DIR"])
    n_classes = len(set(df[target_col]))
    # Create wav paths
    df["wav_path"] = "./data/wavs/" + df["unique_ID"] + ".wav"
    lbl_enc = preprocessing.LabelEncoder()

    # Split train/val/test
    TestSplitter(data_params).get_no_leakage_trainval_test_splits()
    df_train, df_valid, df_test = split_train_val_test(
        df, target_col, seed, data_params["OUTPUT_PATH"]
    )

    print(set(df_train[target_col]))
    print(set(df_valid[target_col]))
    print(set(df_test[target_col]))
    logging.info(
        f"Train: {len(df_train)}, Valid: {len(df_valid)}, Test: {len(df_test)}"
    )
    # Encode target
    train_targets = lbl_enc.fit_transform(df_train[target_col])
    labels = list(sorted(set(df_train[target_col])))
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
        train_dataset, batch_size=14, num_workers=4, pin_memory=True
    )

    wav_augs_eval = wav_aug_combos[hyper_params["wav_augs_eval"]]
    spec_augs_eval = spec_aug_combos[hyper_params["spec_augs_eval"]]
    valid_targets = lbl_enc.fit_transform(df_valid[target_col])
    valid_dataset = ElephantDataset(
        df_valid.wav_path,
        valid_targets,
        params,
        wav_augmentations=wav_augs_eval,
        spec_augmentations=spec_augs_eval,
    )
    valid_dl = torch.utils.data.DataLoader(
        valid_dataset, batch_size=14, shuffle=False, num_workers=4, pin_memory=True
    )

    test_targets = lbl_enc.fit_transform(df_test[target_col])
    test_dataset = ElephantDataset(
        df_test.wav_path,
        test_targets,
        params,
        wav_augmentations=wav_augs_eval,
        spec_augmentations=spec_augs_eval,
    )
    test_dl = torch.utils.data.DataLoader(
        test_dataset, batch_size=14, shuffle=False, num_workers=4, pin_memory=True
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

    engine = Engine(myModel, optimizer, scheduler, loss_fn, device)

    for epoch in range(epochs):
        model_path = f"{seed}-best-model-parameters.pt"
        train_loss = engine.train_one_epoch(train_dl)
        valid_loss, val_targs, val_preds = engine.validate_one_epoch(valid_dl)
        valid_metrics = Metrics(val_targs, val_preds, labels=labels).get_metrics_dict(
            prefix="val"
        )
        test_loss, test_targs, test_preds = engine.validate_one_epoch(test_dl)
        test_metrics = Metrics(test_targs, test_preds, labels=labels).get_metrics_dict(
            prefix="test"
        )

        print(val_targs)
        print(val_preds)
        print(test_targs)
        print(test_preds)
        if neptune_logger is not None:
            neptune_logger[f"train/{seed}/loss"].log(train_loss)
            neptune_logger[f"valid/{seed}/loss"].log(valid_loss)
            neptune_logger[f"test/{seed}/loss"].log(test_loss)

            for k, v in valid_metrics.items():
                neptune_logger[f"valid/{seed}/{k}"].log(v)

            for k, v in test_metrics.items():
                neptune_logger[f"test/{seed}/{k}"].log(v)

        logging.info(
            f"Seed: {seed} ,Training Loss: {train_loss}, Validation Loss: {valid_loss} \nValid F1: {valid_metrics['val_macro avg_f1-score']} Test F1: {test_metrics['test_macro avg_f1-score']}"
        )
        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            min_test_loss = test_loss
            max_valid_f1 = valid_metrics["val_macro avg_f1-score"]
            max_test_f1 = test_metrics["test_macro avg_f1-score"]
            if save_model:
                logging.info("Saving model.....")
                torch.save(myModel.state_dict(), model_path)
                if neptune_logger is not None:
                    neptune_logger[f"best_model_params_{seed}"].track_files(model_path)
        else:
            early_stopping_counter += 1
        if early_stopping_counter > early_stopping_iter:
            logging.info(f"Early stopping after {early_stopping_counter} iterations")
            break

    return min_valid_loss, max_valid_f1, max_test_f1


def train_one_model(
    data_params=data_params,
    best_params=best_params,
    wav_params=wav_aug_combos,
    spec_params=spec_aug_combos,
    save_model=True,
    neptune_logger=None,
):

    val_losses, val_f1, test_f1 = [], [], []

    # Averaging runs over 5 seeds
    for seed in range(100, 600, 100):
        data_params["SEED"] = seed
        temp_loss, temp_f1, temp_test_f1 = run_train(
            data_params,
            best_params,
            wav_params,
            spec_params,
            save_model=save_model,
            neptune_logger=neptune_logger,
        )
        val_losses.append(temp_loss)
        val_f1.append(temp_f1)
        test_f1.append(temp_test_f1)
    mean_loss = np.mean(val_losses)
    mean_val_f1 = np.mean(val_f1)
    mean_test_f1 = np.mean(test_f1)
    logging.info(
        f"Mean Min Loss: {mean_loss}, Mean Max F1 macro {mean_val_f1}, Mean Max Test F1 {mean_test_f1}"
    )
    return mean_loss, mean_val_f1, mean_test_f1


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
        "wav_augs_eval": trial.suggest_categorical(
            "wav_augs_eval", list(wav_aug_combos.keys())
        ),
        "spec_augs_eval": trial.suggest_categorical(
            "spec_augs_eval", list(spec_aug_combos.keys())
        ),
        "num_layers": trial.suggest_int(
            "num_layer", 0, 7
        ),  # additional layers after pretrained models
        "hidden_size": trial.suggest_int("hidden_size", 16, 2048),
        "dropout": trial.suggest_uniform("dropout", 0.1, 0.7),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-6, 1e-2),
        # "target": data_params["STRATIFY_COL"],
        "epochs": 50,
        "batch_size": 14,
    }

    with log_neptune() as run:
        run["parameters"] = hyper_params
        run["mode"] = "search"
        mean_loss, mean_f1, mean_test_f1 = train_one_model(
            best_params=hyper_params, save_model=False, neptune_logger=run
        )

        run["eval/mean_loss"] = mean_loss
        run["eval/mean_f1"] = mean_f1
        run["test/mean_f1"] = mean_test_f1

    return mean_loss


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["search", "train"],
        default="search",
        help="Specify if hyperparameter search should be run, or best_param model should be trained.",
    )

    args = parser.parse_args()

    if args.mode == "search":

        # CrossValidator(
        #    data_params).get_no_leakage_crossval_splits(train_val_indices)

        logging.debug("Hyperparameter search mode")
        # Objective function is to minimize minimum validation loss
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=100)

        # Print and save parameters of best model
        trial_ = study.best_trial
        logging.info(f"Best trial values: {trial_.values}")
        logging.info(f"Best trial values: {trial_.params}")
        p = trial_.params
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
