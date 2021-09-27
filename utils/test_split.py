"""Utilities for consistently splitting out the test data without leakage.

Author: Lucy Tan
"""
import os
from utils import common

# Seed for splitting the dataset into train/val and test. It should be
# hardcoded to ensure the test set is always the same.
_DEFAULT_SEED_FOR_TRAIN_TEST_SPLIT = 22


# Default parameters to use for test splitting if none are provided.
_DEFAULT_PARAMS = {
    'BASE_DATA_DIR': 'dzanga-bai-20210816T230919Z-001/dzanga-bai',
    # Stratify by the 4 category age column to make it more even.
    'STRATIFY_COL': 'age',
    'TEST_SIZE': 0.2,
    'OUTPUT_PATH': 'dzanga-bai-20210816T230919Z-001/dzanga-bai'
}


class TestSplitter:
    """A class to split a dataset into cross validation folds and a test set.
    """

    def __init__(self, params=None):
        """Create a TestSplitter with the given params.
            BASE_DATA_DIR: The path to the directory with the Dzanga Bai data.
            STRATIFY_COL: The column to stratify the data on. The target
                column should be derived from this one.
            TEST_SIZE: The fraction of data to use for testing.
        """
        self._params = _DEFAULT_PARAMS.copy()
        if params is not None:
            self._params.update(params)

    def get_no_leakage_trainval_test_splits(self, save=True):
        """Split the Dzanga Bai data into training/validation and a test set.

        Stratify by STRATIFY_COL and ensure no data leaks by adding rumbles
        with the same id to the same split.

        Output the indices as separate csvs for training/validation and test.
        """
        df = common.load_dz_data(self._params['BASE_DATA_DIR'], target_col=self._params['STRATIFY_COL'])
        stratify_col = self._params['STRATIFY_COL']
        test_size = self._params['TEST_SIZE']
        seed = _DEFAULT_SEED_FOR_TRAIN_TEST_SPLIT
        split_sizes = [1 - test_size, test_size]
        
 
        train_val_indices, test_indices = (
            common.split_and_stratify_without_leakage(
                df, seed, split_sizes, stratify_col))
    
        if save:
            train_val_indices_filename = os.path.join(
                self._params['OUTPUT_PATH'], 'train_val_indices.csv')
            test_indices_filename = os.path.join(
                self._params['OUTPUT_PATH'], 'test_indices.csv')
            common.output_csv(train_val_indices_filename, train_val_indices)
            common.output_csv(test_indices_filename, test_indices)
        return train_val_indices, test_indices


if __name__ == '__main__':
    TestSplitter().get_no_leakage_trainval_test_splits()