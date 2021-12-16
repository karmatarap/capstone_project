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
from .utils import split_train_val_test, log_neptune, set_seeds, seed_worker


def run_train(
    data_params: dict,
    hyper_params: dict,
    wav_aug_combos: dict,
    spec_aug_combos: dict,
    save_model=False,
) -> float:

    # For reproducibility
    g = torch.Generator()
    g.manual_seed(0)

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

    df_train, df_valid, df_test = split_train_val_test(df, data_params)

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
        train_dataset,
        batch_size=14,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
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
        valid_dataset,
        batch_size=14,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
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
        test_dataset,
        batch_size=14,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
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
        train_loss = engine.train_one_epoch(train_dl)
        valid_loss, val_targs, val_preds = engine.validate_one_epoch(valid_dl)
        valid_metrics = Metrics(val_targs, val_preds, labels=labels).get_metrics_dict(
            prefix="val"
        )
        test_loss, test_targs, test_preds = engine.validate_one_epoch(test_dl)
        test_metrics = Metrics(test_targs, test_preds, labels=labels).get_metrics_dict(
            prefix="test"
        )

        logging.debug(
            f"Seed: {seed} ,Training Loss: {train_loss}, Validation Loss: {valid_loss}"
        )
        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            best_valid = valid_metrics
            best_test = test_metrics
            if save_model:
                logging.debug("Saving model.....")
                torch.save(myModel.state_dict(), f"best_model_params_{seed}.pt")

        else:
            early_stopping_counter += 1
        if early_stopping_counter > early_stopping_iter:
            logging.debug(f"Early stopping after {early_stopping_counter} iterations")
            break

    return best_valid, best_test


def train_one_model(
    data_params=data_params,
    best_params=best_params,
    wav_params=wav_aug_combos,
    spec_params=spec_aug_combos,
    save_model=True,
):

    logging.info("Experiment start ========")
    logging.info(f"Params:\n{best_params}")
    for seed in range(100, 400, 100):
        data_params["SEED"] = seed
        data_params["TRAIN_TEST_SPLIT_SEED"] = seed

        # Create train/test split files
        TestSplitter(data_params).get_no_leakage_trainval_test_splits()

        val_metrics, test_metrics = run_train(
            data_params, best_params, wav_params, spec_params, save_model=save_model
        )

        logging.info(
            f"Seed: {seed}, val_metrics:\n{val_metrics}\n, \nTest Metrics \n{test_metrics}"
        )
    logging.info("Experiment End ========")


if __name__ == "__main__":
    logging.basicConfig(filename="best_model.log", level=logging.INFO)

    # Experiment 1, best model
    train_one_model(save_model=True)

    """
    # Experiment 2, 
    for wav in wav_aug_combos:
        #if wav != best_params["wav_augs"]:
        params = best_params
        params["wav_augs"] = wav
        train_one_model(save_model=False, best_params=params)

    for spec in spec_aug_combos:
        #if spec != best_params["spec_augs"]:
        params = best_params
        params["spec_augs"] = spec
        train_one_model(save_model=False, best_params=params)

    """
