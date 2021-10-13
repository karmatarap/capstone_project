import math
import os
from collections import Counter
from typing import Iterable, Iterator, NamedTuple, Tuple

import cv2
import librosa
import neptune.new as neptune
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from audiomentations import (
    AddGaussianNoise,
    AddGaussianSNR,
    Compose,
    Normalize,
    PitchShift,
    SpecFrequencyMask,
)
from efficientnet_pytorch import EfficientNet
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset

from utils.common import load_dz_data
from utils.cross_validation import CrossValidator
from utils.metrics import Metrics
from utils.test_split import TestSplitter

from .config import AudioParams, data_params
from .dataset import ElephantDataset
from .engine import Engine
from .models import get_pretrained_model
from .utils import get_fold, set_seeds


def run_train(
    fold: int,
    data_params: dict,
    hyper_params: dict,
    wav_aug_combos: dict,
    spec_aug_combos: dict,
    save_model=False,
    neptune_run=None,
) -> float:
    print(f"FOLD {fold}")
    print("--------------------------------")

    target_col = data_params["STRATIFY_COL"]
    epochs = hyper_params["epochs"]
    # Load Data
    df = load_dz_data(data_params["BASE_DATA_DIR"], target_col=target_col)

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

    myModel = get_pretrained_model(hyper_params)
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

    # Poor mans early stopping, abort training if validation loss does not
    # improve for x successive epochs where x is 10% of total epochs
    early_stopping_iter = epochs // 10
    early_stopping_counter = 0

    set_seeds(data_params["SEED"])

    engine = Engine(myModel, optimizer, scheduler, loss_fn, device)

    for epoch in range(epochs):

        train_loss = engine.train_one_epoch(train_dl)
        valid_loss, targets, predictions = engine.validate_one_epoch(valid_dl)

        if epochs % 10 == 0:
            neptune_run[f"train/{fold}/loss"].log(train_loss)
            neptune_run[f"valid/{fold}/loss"].log(valid_loss)

            m = Metrics(targets, predictions).score(average_="macro")
            neptune_run[f"valid/{fold}/metrics"].log(m)
            print(
                f"Fold {fold} ,Training Loss: {train_loss}, Validation Loss: {valid_loss}, F1 Score: {m['F1_macro']}"
            )
        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            if save_model:
                print("Saving model.....")
                torch.save(myModel.state_dict(), f"{fold}-best-model-parameters.pt")
        else:
            early_stopping_counter += 1
        if early_stopping_counter > early_stopping_iter:
            break
    return min_valid_loss


# Augmentations to be performed directly on the wav files
# ----------------------------------------------------------------
# Normalize: Add a constant amount of gain, normalizes the loudness
# - Should help normalize if elephants are closer or further from detectors
# PitchShift: Changes pitch without changing tempo
# - Would an elephant running cause the calls to change in pitch?
# AddGuassianNoise: Add gaussian noise
# AddGuassianNoiseSNR: Add gaussian noise with random Signal to Noise Ratio
# SpecFrequencyMask: Mask a set of frequencies: see Google AI SpecAugment

# try these combos
# passing mapping as dicts to allow for logging
wav_aug_combos = {
    "none": None,
    "Norm": Normalize(),
    "Norm-SNR": Compose(
        [Normalize(), AddGaussianSNR(min_snr_in_db=0.0, max_snr_in_db=60.0)]
    ),
    "Norm-Gauss-SNR": Compose(
        [
            Normalize(),
            AddGaussianNoise(),
            AddGaussianSNR(min_snr_in_db=0.0, max_snr_in_db=60.0),
        ]
    ),
    "Norm-Gauss-SNR-Pitch": Compose(
        [
            Normalize(),
            AddGaussianNoise(),
            AddGaussianSNR(min_snr_in_db=0.0, max_snr_in_db=60.0),
            PitchShift(),
        ]
    ),
}

spec_aug_combos = {"none": None, "SpecAug": SpecFrequencyMask()}


def objective(trial):
    # Logging to neptune
    run = neptune.init(
        project=os.getenv("NEPTUNE_PROJECT"), api_token=os.getenv("NEPTUNE_API_TOKEN")
    )

    hyper_params = {
        "pretrained_model": trial.suggest_categorical(
            "pretrained_model", ["resnext50_32x4d", "resnet50", "efficientnet-b4"]
        ),
        "wav_augs": trial.suggest_categorical("wav_augs", list(wav_aug_combos.keys())),
        "spec_augs": trial.suggest_categorical(
            "spec_augs", list(spec_aug_combos.keys())
        ),
        ""
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

    run["parameters"] = hyper_params
    all_losses = []
    for f in range(data_params["NUM_K_FOLDS"]):
        temp_loss = run_train(
            f,
            data_params,
            hyper_params,
            wav_aug_combos,
            spec_aug_combos,
            save_model=False,
            neptune_run=run,
        )
        all_losses.append(temp_loss)
    mean_loss = np.mean(all_losses)
    run["eval/mean_loss"] = mean_loss
    run.stop()
    return mean_loss


if __name__ == "__main__":

    # Objective function is to minimize minimum validation loss
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)

    # Print and save parameters of best model
    print("best trial:")
    trial_ = study.best_trial
    print(trial_.values)
    print(trial_.params)

    scores = 0
    for j in range(data_params["NUM_K_FOLDS"]):
        scr = run_train(
            j, trial_.params, wav_aug_combos, spec_aug_combos, save_model=True
        )
        scores += scr
    print(scores / data_params["NUM_K_FOLDS"])

