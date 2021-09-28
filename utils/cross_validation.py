"""Utilities for splitting the data for k-fold cross validation.

Author: Lucy Tan
"""
import os
from typing import NamedTuple
import numpy as np
from utils import common

# Default parameters to use for cross validation splitting if none are
# provided.
_DEFAULT_PARAMS = {
    "BASE_DATA_DIR": "dzanga-bai-20210816T230919Z-001/dzanga-bai",
    "NUM_K_FOLDS": 5,
    "SEED": 100,
    # Stratify by the 4 category age column to make it more even.
    "STRATIFY_COL": "age",
    "OUTPUT_PATH": "dzanga-bai-20210816T230919Z-001/dzanga-bai",
}


class CrossValidator:
    """A class to split a dataset into cross validation folds.
    """

    def __init__(self, params=None):
        """Create a CrossValidator with the given params.
            BASE_DATA_DIR: The path to the directory with the Dzanga Bai data.
            NUM_K_FOLDS: The number of cross validation folds to use.
            SEED: The random seed to use for splitting the train/val data into
                cross validation folds.
            STRATIFY_COL: The column to stratify the data on. The target
                column should be derived from this one.
        """
        self._params = _DEFAULT_PARAMS.copy()
        if params is not None:
            self._params.update(params)

    def get_no_leakage_crossval_splits(self, train_val_indices):
        """Split the Dzanga Bai data into cross validation folds.

        Stratify by STRATIFY_COL and ensure no data leaks by adding rumbles
        with the same id to the same split.

        Output the indices as separate train and validation csvs for each
        fold.
        """
        df = common.load_dz_data(
            self._params["BASE_DATA_DIR"], target_col=self._params["STRATIFY_COL"]
        )
        num_folds = self._params["NUM_K_FOLDS"]
        train_val_df = df.iloc[train_val_indices]

        # Split the train/val set into NUM_K_FOLDS different folds with a seed
        # that can change between runs to get different cross validation
        # splits for different trials.
        stratify_col = self._params["STRATIFY_COL"]
        split_sizes = [len(train_val_indices) / num_folds] * num_folds
        split_indices = common.split_and_stratify_without_leakage(
            train_val_df, self._params["SEED"], split_sizes, stratify_col
        )
        # Test indices are assumed to be all other indices.
        test_indices = set(range(len(df))) - set(train_val_indices)
        for i in range(len(split_indices)):
            # Train is all splits except the current split.
            # Validation is the current split.
            # These indices are relative to the train_val_df.
            # They are later converted to be of the original df.
            train_indices_old = np.concatenate(
                split_indices[:i] + split_indices[i + 1 :]
            ).ravel()
            val_indices_old = split_indices[i]
            # Get the iloc indices for train and val in the original df.
            train_idx_original_indices = set(
                train_val_df.iloc[train_indices_old].index.values
            )
            val_idx_original_indices = set(
                train_val_df.iloc[val_indices_old].index.values
            )
            train_idx = df.reset_index()[
                df.reset_index()["index"].apply(
                    lambda x: x in train_idx_original_indices
                )
            ].index
            val_idx = df.reset_index()[
                df.reset_index()["index"].apply(lambda x: x in val_idx_original_indices)
            ].index
            # For each fold, there should be no overlap between train, val,
            # and test.
            assert len(set(train_idx) | set(val_idx) | test_indices) == len(
                train_idx
            ) + len(val_idx) + len(test_indices)
            train_indices_filename = os.path.join(
                self._params["OUTPUT_PATH"], f"train_indices_{i}.csv"
            )
            val_indices_filename = os.path.join(
                self._params["OUTPUT_PATH"], f"val_indices_{i}.csv"
            )
            common.output_csv(train_indices_filename, train_idx)
            common.output_csv(val_indices_filename, val_idx)


if __name__ == "__main__":
    train_val_indices_filename = os.path.join(
        _DEFAULT_PARAMS["OUTPUT_PATH"], "train_val_indices.csv"
    )
    with open(train_val_indices_filename, "rt") as f:
        train_val_indices = np.array([int(index) for index in f.readlines()])
    CrossValidator().get_no_leakage_crossval_splits(train_val_indices)
