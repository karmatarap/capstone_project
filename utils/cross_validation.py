"""Utilities for splitting the data for k-fold cross validation.

Author: Lucy Tan

This module exports the CrossValidator class for splitting the dataset into
cross validation folds. The split is stratified by the target column.
Additionally, it is guaranteed to not have any rumbles with the same id (i.e.
produced by the same elephant in short succession) in the validation sets as
in the training data. Thus it does not have any leakage.
"""
import os
from typing import NamedTuple

import numpy as np

from utils import common


# Default parameters to use for cross validation splitting if none are
# provided.
_DEFAULT_PARAMS = {
    'BASE_DATA_DIR': 'dzanga-bai-20210816T230919Z-001/dzanga-bai',
    'BASE_ELP_WAV_DATA_DIR': 'elp_data/wav_files',
    'NUM_K_FOLDS': 5,
    # Takes precedence over NUM_K_FOLDS, creating 1 fold with train as
    # 1-VAL_SIZE and val as VAL_SIZE.
    'VAL_SIZE': None,
    'SEED': 100,
    # Stratify by the 4 category age column to make it more even.
    'STRATIFY_COL': 'age',
    'OUTPUT_PATH': 'dzanga-bai-20210816T230919Z-001/foo',
}


class CrossValidator:
    """A class to split a dataset into cross validation folds.

    The public method is get_no_leakage_crossval_splits, which outputs csvs
    of the cross validation folds and the test set for the Dzanga Bai data.
    It uses stratification and prevents data leakage by ensuring rumbles with
    the same id end up in the same split.
    """

    def __init__(self, params=None):
        """Create a CrossValidator with the given params.

        The params should include all of the following:
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


    def get_no_leakage_crossval_splits(self, train_val_indices, save=True):
        """Split the Dzanga Bai data into cross validation folds.

        Stratify by STRATIFY_COL and ensure no data leaks by adding rumbles
        with the same id to the same split.

        Output the indices as separate train and validation csvs for each
        fold.
        """
        df = common.load_dz_data(self._params['BASE_DATA_DIR'])
        stratify_col = self._params['STRATIFY_COL']
        train_val_df = df.iloc[train_val_indices]
        val_size = self._params['VAL_SIZE']
        if val_size is not None:
            num_folds = 1
            # First is val, second is train.
            split_sizes  = [
                len(train_val_indices) * val_size,
                len(train_val_indices) * (1 - val_size)
            ]
        else:
            num_folds = self._params['NUM_K_FOLDS']
            split_sizes = [len(train_val_indices) / num_folds] * num_folds

        # Split the train/val set into NUM_K_FOLDS different folds with a seed
        # that can change between runs to get different cross validation
        # splits for different trials.
        split_indices = common.split_and_stratify_without_leakage(
            train_val_df, self._params['SEED'], split_sizes, stratify_col)
        # Test indices are assumed to be all other indices.
        test_indices = set(range(len(df))) - set(train_val_indices)
        cross_val_indices = []
        for i in range(num_folds):
            # Train is all splits except the current split.
            # Validation is the current split.
            # These indices are relative to the train_val_df.
            # They are later converted to be of the original df.
            train_indices_old = np.concatenate(
                split_indices[:i] + split_indices[i+1:]).ravel()
            val_indices_old = split_indices[i]
            # Get the iloc indices for train and val in the original df.
            train_idx_original_indices = set(
                train_val_df.iloc[train_indices_old].index.values)
            val_idx_original_indices = set(
                train_val_df.iloc[val_indices_old].index.values)
            train_idx = df.reset_index()[df.reset_index()['index'].apply(
                lambda x: x in train_idx_original_indices)].index
            val_idx = df.reset_index()[df.reset_index()['index'].apply(
                lambda x: x in val_idx_original_indices)].index
            # For each fold, there should be no overlap between train, val,
            # and test.
            assert(len(set(train_idx) | set(val_idx) | test_indices)
                == len(train_idx) + len(val_idx) + len(test_indices))
            if save:
                train_indices_filename = os.path.join(
                    self._params['OUTPUT_PATH'], f'train_indices_{i}.csv')
                val_indices_filename = os.path.join(
                    self._params['OUTPUT_PATH'], f'val_indices_{i}.csv')
                common.output_csv(train_indices_filename, train_idx)
                common.output_csv(val_indices_filename, val_idx)
            cross_val_indices.append((train_idx, val_idx))
        return cross_val_indices



    def get_no_leakage_crossval_elp_splits(self):
        dataset = common.load_elp_data(self._params['BASE_ELP_WAV_DATA_DIR'])
        num_folds = self._params['NUM_K_FOLDS']
        shuffled_indices = np.random.default_rng(seed=self._params['SEED']).permutation(len(dataset))
        split_indices = np.array_split(shuffled_indices, num_folds)
        for i in range(num_folds):
            train_indices = np.concatenate(
                split_indices[:i] + split_indices[i+1:]).ravel()
            val_indices = split_indices[i]
            train_indices_filename = os.path.join(
                self._params['OUTPUT_PATH'], f'elp_train_indices_{i}.csv')
            val_indices_filename = os.path.join(
                self._params['OUTPUT_PATH'], f'elp_val_indices_{i}.csv')
            common.output_csv(train_indices_filename, train_indices)
            common.output_csv(val_indices_filename, val_indices)



if __name__ == '__main__':
    train_val_indices_filename = os.path.join(
        _DEFAULT_PARAMS['OUTPUT_PATH'], 'train_val_indices.csv')
    with open(train_val_indices_filename, 'rt') as f:
        train_val_indices = np.array([int(index) for index in f.readlines()])
    CrossValidator({'VAL_SIZE': 0.2}).get_no_leakage_crossval_splits(train_val_indices)
    # CrossValidator().get_no_leakage_crossval_elp_splits()