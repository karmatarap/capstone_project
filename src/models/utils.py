import os
import random
from typing import Tuple
from contextlib import contextmanager
import neptune.new as neptune
import numpy as np
import pandas as pd
import torch


def set_seeds(seed: int) -> None:
    """ Setting seeds for all possible random number generators"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_fold(df, fold: int, output_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Return training and validation datasets for a particular fold """
    with open(os.path.join(output_path, f"train_indices_{fold}.csv"), "rt") as f:
        train_indices = np.array([int(index) for index in f.readlines()])
    with open(os.path.join(output_path, f"val_indices_{fold}.csv"), "rt") as f:
        val_indices = np.array([int(index) for index in f.readlines()])

    df_train = df[df.index.isin(train_indices)].reset_index(drop=True)
    df_val = df[df.index.isin(train_indices)].reset_index(drop=True)

    return df_train, df_val


def get_test_indices(output_path: str) -> np.array:
    """Helper function to read test indices"""
    test_indices_filename = os.path.join(output_path, "test_indices.csv")
    with open(test_indices_filename, "rt") as f:
        test_indices = np.array([int(index) for index in f.readlines()])
    return test_indices


@contextmanager
def log_neptune():
    run = neptune.init(
        project=os.getenv("NEPTUNE_PROJECT"), api_token=os.getenv("NEPTUNE_API_TOKEN")
    )
    try:
        yield run
    finally:
        run.stop()
