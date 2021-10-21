"""Utilities for augmenting the data and generating batches of it.

Author: Lucy Tan

This module exports the Augmenter class for augmenting the training data with
image augmentation techniques, balancing the classes of the training data, and
generating sequences that yield batches of data. The sequences are designed
for use with the Keras APIs for model training, evaluating, and prediction.
"""
import glob
import math
import os
import shelve
from typing import Sequence, NamedTuple, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio

from utils import common


# Default parameters to use for augmenting if none are provided.
_DEFAULT_PARAMS = {
    'AUGMENTATION_ARGS': dict(width_shift_range=0.3,
            horizontal_flip=True,
            brightness_range=[0.3, 0.9],
            fill_mode='nearest'),
    'BASE_DATA_DIR': 'dzanga-bai-20210816T230919Z-001/dzanga-bai',
    'BASE_ELP_WAV_DATA_DIR': 'elp_data/wav_files',
    'OUTPUT_PATH': 'dzanga-bai-20210816T230919Z-001/dzanga-bai',
    'SHELF_PATH': 'backups/shelves/shelf-%s-%s-%d',
    'BATCH_SIZE': 16,
    'IMAGE_SIZE': 512,
    'SEED': 100,
    'TARGET_COL': 'agecat',
    'USE_WAV_FILES': False,
    'AUGMENT_WITH_BALANCED_CLASSES': False,
}



# Adapted from https://stackoverflow.com/questions/66410340/filter-tensorflow-dataset-by-id
def filter_elp_dataset(dataset, indices):
    keys_tensor = tf.constant(indices)
    vals_tensor = tf.ones_like(keys_tensor)  # Ones will be casted to True.

    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor),
        default_value=0)  # If index not in table, return 0.

    @tf.function
    def hash_table_filter(index, value):
        table_value = table.lookup(index)  # 1 if index in arr, else 0.
        index_in_arr =  tf.cast(table_value, tf.bool) # 1 -> True, 0 -> False
        return index_in_arr

    return dataset.enumerate().filter(hash_table_filter).map(lambda idx, value: value)


# Copied from https://www.tensorflow.org/tutorials/audio/transfer_learning_audio
@tf.function
def load_wav_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
          file_contents,
          desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return tf.reshape(wav, [-1])


class CrossValFoldSequence(NamedTuple):
    """Contains the sequences for a single cross validation fold.

    Each batch is a tuple of x_data, y_data, where x_data is spectrogram pixel
    values and y_data is 1-hot encoded target labels.
    
    train: The sequence used for the train data of this fold.
    val: The sequence used for the validation data of this fold.
    """
    train: tf.keras.utils.Sequence
    val: tf.keras.utils.Sequence


class DatasetSequence(tf.keras.utils.Sequence):

    def __init__(self, dataset, labels, label_mapping, shelf_path, length=None):
        self._dataset = dataset
        self.classes = labels
        self.class_indices = label_mapping
        if length is None:
            self._length = len(dataset)
        else:
            self._length = length
        self._shelf_path = shelf_path
        self._data = CompressedShelf(shelf_path)

    def __len__(self):
        return self._length

    def _prefetch_data(self):
        dataset_iterator = self._dataset.as_numpy_iterator()
        for i in range(self._length):
            self._data[str(i)] = next(dataset_iterator)
        self._data.sync()
    
    def __getitem__(self, index):
        str_index = str(index)
        if str_index not in self._data:
            print('Prefetching data')
            self._prefetch_data()
        return self._data[str_index]

    def delete_data(self):
        self._data.close()
        try:
            for f in glob.glob(f'{self._shelf_path}*'):
                os.remove(f)
                print('Deleted', f)
        except Exception as e:
            print('Failed to delete shelf')
            print(e)

    def on_epoch_end(self):
        pass


class CombinedSequence(tf.keras.utils.Sequence):
    """Combines sequences to produce data from all the given sequences.

    It is not guaranteed that each batch will have the same amount of data
    from each given sequence. Instead, if there are N sequences, then across
    each group of N batches, there will be batch_size elements from each
    sequence.

    It is designed to be used with the Keras APIs for model training,
    evaluating, and prediction. As a result, it needs to implement the
    following methods:
        __len__: indicates how many total elements are in the sequence
        __iter__: iterates through the sequence, generating each batch of data
        __next__: yields the next batch of data
    """
    def __init__(self, seed, batch_size, labels, label_mapping, *sequences):
        """Create a CombinedSequence with the seed, batch size, and sequences.

        The sequence is created to use in the __next__ method.
        """
        self._seed = seed
        self._batch_size = batch_size
        self._sequences = sequences
        assert batch_size % len(sequences) == 0, 'Batch size must be a multiple of the number of sequences to make each batch even'
        self._rng = np.random.default_rng(seed=self._seed)
        self._num_batches_per_sequence = self._get_num_batches_per_sequence()
        self._len = max(self._num_batches_per_sequence) * len(sequences)
        self.classes = []
        for _ in range(self._len):
            for i in range(len(sequences)):
                self.classes.extend([labels[i]] * (batch_size // len(sequences)))
        self.class_indices = label_mapping
        
    def __len__(self):
        return self._len

    def _get_num_batches_per_sequence(self):
        full_sequence_lengths = []
        for seq in self._sequences:
            i = len(seq) - 1
            while i > -1:
                if len(seq[i][0]) >= self._batch_size:
                    full_sequence_lengths.append(i + 1)
                    break
                i -= 1
            if i == -1:
                full_sequence_lengths.append(1)
        return full_sequence_lengths


    def _batches_per_sequence(self):
        shortest_length = len(min(self._sequences, key=len))
        while shortest_length > 0:
            all_are_valid = True
            for sequence in self._sequences:
                if len(sequence[shortest_length - 1][0]) < self._batch_size:
                    all_are_valid = False
                    break
            if all_are_valid:
                break
            shortest_length -= 1
        return shortest_length

    def _get_data_for_epoch(self):
        x_data = []
        y_data = []
        for sequence in self._sequences:
            for i in range(len(sequence)):
                cur_x_data, cur_y_data = sequence[i]
                # Don't use batch groups where a sequence had less than a full
                # batch. This would allow the training data to be unbalanced.
                if len(cur_x_data) != self._batch_size:
                    continue
                x_data.append(cur_x_data)
                y_data.append(cur_y_data)
        merged_data = np.concatenate(x_data), np.concatenate(y_data)
        data_len = len(merged_data[0])
        indices = np.arange(data_len)
        self._rng.shuffle(indices)
        return merged_data[0][indices], merged_data[1][indices]
    
    def __getitem__(self, index):
        sequence_batch_index = index // len(self._sequences)
        sequence_batches = [self._sequences[i][sequence_batch_index % self._num_batches_per_sequence[i]] for i in range(len(self._sequences))]
        batch_start_index = self._batch_size * (index % len(self._sequences))
        batch_end_index = batch_start_index + self._batch_size
        batch_indices = np.arange(batch_start_index, batch_end_index)
        x_data = []
        y_data = []
        for i in range(len(self._sequences)):
            cur_indices = batch_indices[batch_indices % len(self._sequences) == i] // len(self._sequences)
            x_data.extend(sequence_batches[i][0][cur_indices])
            y_data.extend(sequence_batches[i][1][cur_indices])
        indices = np.arange(len(x_data))
        return np.array(x_data)[indices], np.array(y_data)[indices]

    def on_epoch_end(self):
        for sequence in self._sequences:
            sequence.on_epoch_end()
        self._rng = np.random.default_rng(seed=self._seed)
        

class Augmenter:
    """A class to augment the training data and create sequences of batches.

    The public methods are:
        get_sequences: Returns the cross validation train and val sequences
            for the Dzanga Bai data given the indices (output by the
            CrossValidator).
        get_test_sequence: Returns the test sequence for the Dzanga Bai data
            given the indices (output by the TestSplitter).
    """

    def __init__(self, params=None):
        """Create a Augmenter with the given params.

        The params should include all of the following:
            AUGMENTATION_ARGS
            BASE_DATA_DIR: The path to the directory with the Dzanga Bai data.
            BATCH_SIZE: The number of data points to use in each batch.
            IMAGE_SIZE: The size in pixels for both height and width of the
                spectrogram when input to the model.
            SEED: The random seed to use for consistent image augmentation and
                data shuffling.
            TARGET_COL: The name of the column containing the target label.
        """
        self._params = _DEFAULT_PARAMS.copy()
        if params is not None:
            self._params.update(params)

    def _load_wav_for_map(self, filename, one_hot_label, label):
        return load_wav_16k_mono(filename), one_hot_label, label

    def _get_sequence_audio(self, df, filtered_df, shuffle, fold_index, is_train_fold):
        target_col = self._params['TARGET_COL']
        one_hot_labels = pd.get_dummies(filtered_df[target_col].astype(pd.CategoricalDtype(categories=sorted(list(df[target_col].unique())))))
        dataset = tf.data.Dataset.from_tensor_slices((filtered_df['wav_path'], one_hot_labels, np.argmax(one_hot_labels.to_numpy(), axis=1)))
        dataset = dataset.map(self._load_wav_for_map)
        max_length = dataset.reduce(np.int32(0), lambda cur_max, new_x: tf.math.maximum(cur_max, tf.shape(new_x[0])[0])).numpy()
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(filtered_df), seed=self._params['SEED'], reshuffle_each_iteration=False)
        dataset_labels = list(dataset.map(lambda x,y,label: label).as_numpy_iterator())
        label_mapping = {value: index for index, value in enumerate(one_hot_labels.columns)}
        dataset = dataset.map(lambda x,y,label: (tf.reshape(tf.pad(x, [[0, max_length - tf.shape(x)[0]]]), [max_length]),y))
        shelf_path = self._params['SHELF_PATH'] % ('dz', 'train' if is_train_fold else ('val' if fold_index is not None else 'test'), fold_index if fold_index is not None else 0)
        return DatasetSequence(dataset.batch(self._params['BATCH_SIZE']).prefetch(tf.data.AUTOTUNE), dataset_labels, label_mapping, shelf_path)

    def _get_sequence_image(self, df, filtered_df, shuffle, augment):
        target_col = self._params['TARGET_COL']
        image_size = self._params['IMAGE_SIZE']
        augmentation_args = self._params['AUGMENTATION_ARGS'] if augment else {}
        return tf.keras.preprocessing.image.ImageDataGenerator(
            **augmentation_args).flow_from_dataframe(
            dataframe=filtered_df,
            x_col='path',
            y_col=target_col,
            target_size=(image_size, image_size),
            color_mode='rgb',
            classes=sorted(list(df[target_col].unique())),
            class_mode='categorical',
            batch_size=self._params['BATCH_SIZE'],
            shuffle=shuffle,
            seed=self._params['SEED'],
            save_to_dir=None,
            save_prefix='',
            save_format='png',
            subset=None,
            interpolation='nearest',
            validate_filenames=True,)

    def _get_sequence(self, df, shuffle, augment, fold_index, is_train_fold, indices=None):
        filtered_df = df.iloc[indices] if indices is not None else df
        if self._params['USE_WAV_FILES']:
            return self._get_sequence_audio(df, filtered_df, shuffle, fold_index, is_train_fold)
        return self._get_sequence_image(df, filtered_df, augment, shuffle)

    def _get_elp_sequence(self, dataset, shuffle, indices, fold_index, is_train_fold):
        dataset = filter_elp_dataset(dataset, indices)
        max_length = dataset.reduce(np.int32(0), lambda cur_max, new_x: tf.math.maximum(cur_max, tf.shape(new_x)[0])).numpy()
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(indices), seed=self._params['SEED'], reshuffle_each_iteration=False)
        dataset = dataset.map(lambda x: tf.reshape(tf.pad(x, [[0, max_length - tf.shape(x)[0]]]), [max_length]))
        dataset = dataset.map(lambda x: (x, np.zeros(())))
        shelf_path = self._params['SHELF_PATH'] % ('elp', 'train' if is_train_fold else 'val', fold_index)
        return DatasetSequence(dataset.batch(self._params['BATCH_SIZE']).prefetch(tf.data.AUTOTUNE), None, None, shelf_path, math.ceil(len(indices) / self._params['BATCH_SIZE']))

    def _filter_indices(self, df, indices, value_to_accept):
        new_indices = [
            idx for idx in indices
            if df[self._params['TARGET_COL']].iloc[idx] == value_to_accept
        ]
        return np.array(new_indices)

    def _get_balanced_sequence(self, df, indices, fold_index):
        seed = self._params['SEED']
        target_col = self._params['TARGET_COL']
        sequences = []
        for target_val in sorted(df[target_col].unique()):
            filtered_indices = self._filter_indices(
                df, indices, target_val)
            sequences.append(self._get_sequence(
                df, shuffle=True, augment=True, fold_index=fold_index, is_train_fold=True, indices=filtered_indices))
        label_mapping = {value: index for index, value in enumerate(sorted(list(df[target_col].unique())))}
        return CombinedSequence(seed, self._params['BATCH_SIZE'], list(range(len(sequences))), label_mapping, *sequences)

    def get_test_sequence(self, test_indices):
        df = common.load_dz_data(self._params['BASE_DATA_DIR'])
        return self._get_sequence(df, shuffle=False, augment=False, fold_index=None, is_train_fold=False, indices=test_indices)


    def get_sequences(self, cross_val_indices, augment=True):
        """Get the sequences for the given indices.

        If augment is True, use image augmentation and class balancing for the
        training data.

        For each cross-validation fold, generate a sequence for the train and
        validation data.
        """
        df = common.load_dz_data(self._params['BASE_DATA_DIR'])
        target_col = self._params['TARGET_COL']
        sequences = []
        for i, fold_indices in enumerate(cross_val_indices):
            if augment and self._params['AUGMENT_WITH_BALANCED_CLASSES'] and not self._params['USE_WAV_FILES']:
                train_sequence = self._get_balanced_sequence(
                    df, fold_indices[0], i)
            else:
                train_sequence = self._get_sequence(
                    df, shuffle=augment, augment=augment, fold_index=i, is_train_fold=True, indices=fold_indices[0])
            val_sequence = self._get_sequence(
                df, shuffle=False, augment=False, fold_index=i, is_train_fold=False, indices=fold_indices[1])
            sequences.append(CrossValFoldSequence(
                train_sequence, val_sequence))
        return sequences


    def get_elp_sequences(self, cross_val_indices, shuffle=True):
        """Get the ELP sequences for the given indices.

        For each cross-validation fold, generate a sequence for the train and
        validation data.
        """
        dataset = common.load_elp_data(self._params['BASE_ELP_WAV_DATA_DIR'])
        sequences = []
        for i, fold_indices in enumerate(cross_val_indices):
            train_sequence = self._get_elp_sequence(
                    dataset, shuffle=shuffle, fold_index=i, is_train_fold=True, indices=fold_indices[0])
            val_sequence = self._get_elp_sequence(
                dataset, shuffle=False, fold_index=i, is_train_fold=False, indices=fold_indices[1])
            sequences.append(CrossValFoldSequence(
                train_sequence, val_sequence))
        return sequences


if __name__ == '__main__':
    # train_indices_filename_pattern = os.path.join(
    #     _DEFAULT_PARAMS['OUTPUT_PATH'], 'train_indices_%s.csv')
    # val_indices_filename_pattern = os.path.join(
    #     _DEFAULT_PARAMS['OUTPUT_PATH'], 'val_indices_%s.csv')
    # cross_val_indices = []
    # for i in range(5):
    #     with open(train_indices_filename_pattern % i, 'rt') as f:
    #         train_indices = np.array([int(index) for index in f.readlines()])
    #     with open(val_indices_filename_pattern % i, 'rt') as f:
    #         val_indices = np.array([int(index) for index in f.readlines()])
    #     cross_val_indices.append((train_indices,val_indices))
    # augmenter = Augmenter()
    # kfold_sequences = augmenter.get_sequences(cross_val_indices)
    # test_indices_filename = os.path.join(
    #     _DEFAULT_PARAMS['BASE_DATA_DIR'], 'test_indices.csv')
    # with open(test_indices_filename, 'rt') as f:
    #     test_indices = np.array([int(index) for index in f.readlines()])
    # test_sequence = augmenter.get_test_sequence(test_indices)
    # cross_val_data = [
    #     (next(iter(sequence.train)), next(iter(sequence.val)))
    #     for sequence in kfold_sequences
    # ]
    # #print(cross_val_data, next(iter(test_sequence)))
    # #print(test_sequence.classes, test_sequence.class_indices)
    # print(kfold_sequences[0].train.classes)
    # print(kfold_sequences[0].train.class_indices)
    # for i in range(len(kfold_sequences[0].train)):
    #     print(kfold_sequences[0].train[i][1])


    train_indices_filename_pattern = os.path.join(
        _DEFAULT_PARAMS['OUTPUT_PATH'], 'elp_train_indices_%s.csv')
    val_indices_filename_pattern = os.path.join(
        _DEFAULT_PARAMS['OUTPUT_PATH'], 'elp_val_indices_%s.csv')
    cross_val_indices = []
    for i in range(5):
        with open(train_indices_filename_pattern % i, 'rt') as f:
            train_indices = np.array([int(index) for index in f.readlines()])
        with open(val_indices_filename_pattern % i, 'rt') as f:
            val_indices = np.array([int(index) for index in f.readlines()])
        cross_val_indices.append((train_indices,val_indices))
    augmenter = Augmenter()
    kfold_sequences = augmenter.get_elp_sequences(cross_val_indices)
    cross_val_data = [
        (sequence.train[0], sequence.val[0])
        for sequence in kfold_sequences
    ]
    cross_val_len = [
        (len(sequence.train), len(sequence.val))
        for sequence in kfold_sequences
    ]
    print(cross_val_data, cross_val_len)