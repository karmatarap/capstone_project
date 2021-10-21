"""Utilities for generating triplets for representation learning.

Author: Lucy Tan

This module exports the TripletGenerator classes for generating triplets of spectrograms for use in representation learning.

Triplets are comprised of 3 elements: anchor, positive, and negative. The anchor is the base spectrogram to which the positive and negative are compared. The positive is the spectrogram that should be closer to the anchor, and the negative is the spectrogram that should be further from the anchor. For example, if the anchor is represented by 0, it is good if the positive is represented as 2, while the negative is represented by 6.

There are 5 different TripletGenerator subclasses, each corresponding to a method from https://arxiv.org/pdf/1711.02209.pdf.
"""
import collections
import os
import shelve

import numpy as np
from scipy.spatial import distance
import tensorflow as tf

from utils import augment
from utils import common


# Default parameters to use for triplet generating if none are provided.
_DEFAULT_PARAMS = {
    'BASE_DATA_DIR': '../dzanga-bai-20210816T230919Z-001/dzanga-bai',
    'OUTPUT_PATH': './dzanga-bai-20210816T230919Z-001/foo',
    'SEED': 100,
    'TRIPLET_GAUSSIAN_STD': 0.03 * 256,
    'TRIPLET_EXAMPLE_MIXING_NEGATIVE_RATIO': 0.25,
    'SEMI_HARD_NEGATIVE_MINING': True,
    'REPRESENTATION_DISTANCE_METRIC': 'euclidean',
    'TRIPLET_GENERATOR_CLASS': 'JointTraining',
    'TRIPLET_GENERATOR_CLASS_RATIOS': {
        # 'GaussianNoise': 0.25,
        # 'TimeTranslation': 0.25,
        # 'ExampleMixing': 0.25,
        # 'ExplicitlyLabeled': 0.25
        'GaussianNoise': 0.34,
        'TimeTranslation': 0.33,
        'ExampleMixing': 0.33,
    },
    'ALL_TRIPLETS_PER_ANCHOR': True,
    'REPRESENTATION_SIZE': 32
}


class TripletGenerator(tf.keras.utils.Sequence):

    def __init__(self, base_sequence, params, encoder_func, all_x_data=None, all_label_ids=None, all_indices_by_label_id=None, all_x_indices_by_data=None):
        self._params = _DEFAULT_PARAMS.copy()
        if params is not None:
            self._params.update(params)
        self._rng = np.random.default_rng(np.random.SFC64(self._params['SEED']))
        self._base_sequence = base_sequence
        self._all_x_data = all_x_data or []
        self._all_label_ids = all_label_ids or []
        self._all_indices_by_label_id = all_indices_by_label_id or collections.defaultdict(set)
        self._all_x_indices_by_data = all_x_indices_by_data or {}
        if not all_x_data:
            label_mapping = {}
            next_label_id = 0
            for i in range(len(self._base_sequence)):
                x_data, y_data = self._base_sequence[i]
                for cur_x_data, cur_y_data in zip(x_data, y_data):
                    index = len(self._all_x_data)
                    self._all_x_indices_by_data[hash(cur_x_data.tobytes())] = index
                    label = cur_y_data.tobytes()
                    if label not in label_mapping:
                        label_mapping[label] = next_label_id
                        next_label_id += 1
                    label_id = label_mapping[label]
                    self._all_indices_by_label_id[label_id].add(index)
                    self._all_x_data.append(cur_x_data)
                    self._all_label_ids.append(label_id)
        self._encoder_func = encoder_func
        
    def __len__(self):
        return len(self._base_sequence)

    def __getitem__(self, index):
        anchors, _ = self._base_sequence[index]
        # anchors = anchors.numpy()
        positives, negatives = self._get_positives_and_negatives(anchors)
        if not self._params['USE_WAV_FILES']:
            positives = np.clip(positives, 0., 255.)
        if (self._params['SEMI_HARD_NEGATIVE_MINING'] and
            self._supports_semi_hard_negative_mining()):
            negatives = self._mine_semi_hard_negatives(anchors, positives, negatives)
        dummy_predictions = np.zeros((len(anchors), self._params['REPRESENTATION_SIZE']))
        return ((anchors, positives, negatives), (dummy_predictions, dummy_predictions, dummy_predictions))

    def _get_positives_and_negatives(self, anchors):
        negatives = self._get_negatives(anchors)
        positives = self._get_positives(anchors, negatives)
        return positives, negatives


    def _get_positives(self, anchors, negatives):
        raise NotImplementedError()

    def _get_negatives(self, anchors):
        negatives = np.zeros_like(anchors)
        for i, anchor in enumerate(anchors):
            index = self._all_x_indices_by_data[hash(anchor.tobytes())]
            non_cur_anchor_indices = list(range(index)) + list(range(index + 1, len(self._all_x_indices_by_data)))
            negative_index = self._rng.choice(non_cur_anchor_indices)
            negatives[i] = self._all_x_data[negative_index]
        return negatives

    def _supports_semi_hard_negative_mining(self):
        return True

    def _mine_semi_hard_negatives(self, anchors, positives, negatives):
        # Adapted from https://github.com/JohnVinyard/experiments/blob/master/unsupervised-semantic-audio-embeddings/within-batch-semi-hard-negative-mining.ipynb
        anchor_embeddings = self._encoder_func(anchors)
        positive_embeddings = self._encoder_func(positives)
        negative_embeddings = self._encoder_func(negatives)
        anchor_to_positive_distances = np.diag(distance.cdist(anchor_embeddings, positive_embeddings, metric=self._params['REPRESENTATION_DISTANCE_METRIC']))
        dist_matrix = distance.cdist(anchor_embeddings, negative_embeddings, metric=self._params['REPRESENTATION_DISTANCE_METRIC'])

        # subtract the anchor-to-positive distances, and clip negative values, 
        # since we don't want to choose negatives that are closer than the 
        # positives
        diff = dist_matrix - anchor_to_positive_distances[:, None]
        FLOAT_MAX = np.finfo(diff.dtype).max
        diff[diff <= 0] = FLOAT_MAX

        # For each anchor, find the negative example that is closest, without
        # being closer than the positive example
        indices = np.argmin(diff, axis=-1)
        return negatives[indices]


class ExplicitlyLabeledTripletGenerator(TripletGenerator):

    def _get_positives(self, anchors, negatives):
        del negatives  # Unused.
        positives = np.zeros_like(anchors)
        for i, anchor in enumerate(anchors):
            index = self._all_x_indices_by_data[hash(anchor.tobytes())]
            cur_label_id = self._all_label_ids[index]
            same_class_indices = self._all_indices_by_label_id[cur_label_id]
            # Don't use the same index for both the anchor and positive.
            valid_indices = same_class_indices - {index}
            positive_index = self._rng.choice(list(valid_indices))
            positives[i] = self._all_x_data[positive_index]
        return positives

    def _get_negatives(self, anchors):
        negatives = np.zeros_like(anchors)
        for i, anchor in enumerate(anchors):
            different_class_indices = []
            index = self._all_x_indices_by_data[hash(anchor.tobytes())]
            cur_label_id = self._all_label_ids[index]
            for label_id, indices in self._all_indices_by_label_id.items():
                if label_id != cur_label_id:
                    different_class_indices.extend(indices)
            negative_index = self._rng.choice(different_class_indices)
            negatives[i] = self._all_x_data[negative_index]
        return negatives

    def _supports_semi_hard_negative_mining(self):
        return False


class GaussianNoiseTripletGenerator(TripletGenerator):

    def _get_positives(self, anchors, negatives):
        del negatives  # Unused.
        return anchors + self._rng.normal(
            0, self._params['TRIPLET_GAUSSIAN_STD'], anchors.shape)


class TimeTranslationTripletGenerator(TripletGenerator):

    def _get_positives(self, anchors, negatives):
        del negatives  # Unused.
        time_axis = 1 if self._params['USE_WAV_FILES'] else 2
        shift = self._rng.integers(0, anchors.shape[time_axis])
        return np.roll(anchors, shift, axis=time_axis)


class ExampleMixingTripletGenerator(TripletGenerator):

    def _get_positives(self, anchors, negatives):
        base_negative_ratio = self._params[
            'TRIPLET_EXAMPLE_MIXING_NEGATIVE_RATIO']
        anchor_axes = tuple(range(len(anchors.shape)))[1:]
        negative_axes = tuple(range(len(negatives.shape)))[1:]
        negative_ratios = base_negative_ratio * (
            np.sum(anchors, axis=anchor_axes) /
            np.sum(negatives, axis=negative_axes))
        result = anchors + np.expand_dims(negative_ratios, negative_axes) * negatives
        return result

    def _supports_semi_hard_negative_mining(self):
        return False


class JointTrainingTripletGenerator(TripletGenerator):

    def __init__(self, base_sequence, params, encoder_func):
        self._params = _DEFAULT_PARAMS.copy()
        if params is not None:
            self._params.update(params)
        super().__init__(base_sequence, params, encoder_func)
        self._triplet_generators = []
        for name in self._params['TRIPLET_GENERATOR_CLASS_RATIOS'].keys():
            self._triplet_generators.append(CreateTripletGenerator(name, base_sequence, params, encoder_func, all_x_data=self._all_x_data, all_label_ids=self._all_label_ids, all_indices_by_label_id=self._all_indices_by_label_id, all_x_indices_by_data=self._all_x_indices_by_data))
        self._indices_for_semi_hard_negative_mining = None
        self._batches = {}

    def on_epoch_end(self):
        self._batches = {}

    def _get_positives_and_negatives(self, anchors):
        triplet_generator_class_ratios = self._params['TRIPLET_GENERATOR_CLASS_RATIOS']
        num_generator_types = len(triplet_generator_class_ratios)
        negatives = np.empty_like(anchors)
        positives = np.empty_like(anchors)
        if self._params['ALL_TRIPLETS_PER_ANCHOR']:
            splits = np.array_split(np.arange(len(anchors)), num_generator_types)
        else:
            permutation = self._rng.permutation(np.arange(len(anchors)))
            chunk_sizes = np.array(list(triplet_generator_class_ratios.values())) * len(anchors)
            splits = np.split(permutation, np.round(np.cumsum(chunk_sizes)).astype(int))
        self._indices_for_semi_hard_negative_mining = []
        for split, triplet_generator in zip(splits, self._triplet_generators):
            self._apply_split(split, triplet_generator, anchors, negatives, positives)
        return positives, negatives

    def _apply_split(self, split, triplet_generator, anchors, negatives, positives):
        if triplet_generator._supports_semi_hard_negative_mining():
            self._indices_for_semi_hard_negative_mining.extend(list(split))
        positives[split], negatives[split] = triplet_generator._get_positives_and_negatives(anchors[split])

    def _mine_semi_hard_negatives(self, anchors, positives, negatives):
        if not self._indices_for_semi_hard_negative_mining:
            return anchors, positives, negatives
        indices_arr = np.array(self._indices_for_semi_hard_negative_mining)
        negatives[indices_arr] = super()._mine_semi_hard_negatives(anchors[indices_arr], positives[indices_arr], negatives[indices_arr])
        return negatives

    def __len__(self):
        if self._params['ALL_TRIPLETS_PER_ANCHOR']:
            return len(self._base_sequence) * len(self._triplet_generators)
        return super().__len__()

    def __getitem__(self, index):
        if not self._params['ALL_TRIPLETS_PER_ANCHOR']:
            return super().__getitem__(index)
        if index in self._batches:
            return self._batches[index]
        num_generator_types = len(self._triplet_generators)
        anchors, _ = self._base_sequence[index // num_generator_types]
        # anchors = anchors.numpy()
        original_data_length = len(anchors)
        tile_shape = (num_generator_types, ) + (1,) * (anchors.ndim - 1)
        anchors = np.tile(anchors, tile_shape)
        positives, negatives = self._get_positives_and_negatives(anchors)
        if not self._params['USE_WAV_FILES']:
            positives = np.clip(positives, 0., 255.)
        if (self._params['SEMI_HARD_NEGATIVE_MINING'] and
            self._supports_semi_hard_negative_mining()):
            negatives = self._mine_semi_hard_negatives(anchors, positives, negatives)
        indices = self._rng.permutation(np.arange(len(anchors)))
        dummy_predictions = np.zeros((original_data_length, self._params['REPRESENTATION_SIZE']))
        anchors, positives, negatives = anchors[indices], positives[indices], negatives[indices]
        min_index = (index // num_generator_types) * num_generator_types
        for i in range(num_generator_types):
            start_index = i * original_data_length
            end_index = start_index + original_data_length
            self._batches[i + min_index] = ((anchors[start_index:end_index], positives[start_index:end_index], negatives[start_index:end_index]), (dummy_predictions, dummy_predictions, dummy_predictions))
        return self._batches[index]


class MultiDatasetTripletGenerator(tf.keras.utils.Sequence):
    def __init__(self, triplet_generators, params=None):
        self._triplet_generators = triplet_generators
        self._params = _DEFAULT_PARAMS.copy()
        if params is not None:
            self._params.update(params)
        self._rng = np.random.default_rng(seed=self._params['SEED'])
        self._generator_lengths = list(map(len, triplet_generators))
        self._len = sum(self._generator_lengths)
        self._index_to_generator_index = []
        self._reset_index_mapping()

    def _reset_index_mapping(self):
        count_by_generator = [0] * len(self._triplet_generators)
        self._index_to_generator_index = []
        for index in self._rng.permutation(self._len):
            cur_generator = 0
            while index >= self._generator_lengths[cur_generator]:
                index -= self._generator_lengths[cur_generator]
                cur_generator += 1
            self._index_to_generator_index.append((cur_generator, count_by_generator[cur_generator]))
            count_by_generator[cur_generator] += 1

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        generator, generator_index = self._index_to_generator_index[index]
        return self._triplet_generators[generator][generator_index]

    def on_epoch_end(self):
        for triplet_generator in self._triplet_generators:
            triplet_generator.on_epoch_end()


def CreateTripletGenerator(name, base_sequence, params, encoder_func, **kwargs):
    class_by_name = {
        'ExplicitlyLabeled': ExplicitlyLabeledTripletGenerator,
        'GaussianNoise': GaussianNoiseTripletGenerator,
        'TimeTranslation': TimeTranslationTripletGenerator,
        'ExampleMixing': ExampleMixingTripletGenerator,
        'JointTraining': JointTrainingTripletGenerator
    }
    return class_by_name[name](base_sequence, params, encoder_func, **kwargs)


if __name__ == '__main__':
    PARAMS = {
        'BASE_DATA_DIR': './dzanga-bai-20210816T230919Z-001/dzanga-bai',
        'OUTPUT_PATH': './dzanga-bai-20210816T230919Z-001/foo',
        'NUM_K_FOLDS': 5,
        'SEED': 100,
        'AUGMENTATION_ARGS': {},
        'USE_WAV_FILES': True
    }
    train_indices_filename_pattern = os.path.join(PARAMS['OUTPUT_PATH'], 'train_indices_%s.csv')
    val_indices_filename_pattern = os.path.join(PARAMS['OUTPUT_PATH'], 'val_indices_%s.csv')
    cross_val_indices = []
    for i in range(PARAMS['NUM_K_FOLDS']):
        with open(train_indices_filename_pattern % i, 'rt') as f:
            train_indices = np.array([int(index) for index in f.readlines()])
        with open(val_indices_filename_pattern % i, 'rt') as f:
            val_indices = np.array([int(index) for index in f.readlines()])
        cross_val_indices.append((train_indices,val_indices))
    test_indices_filename = os.path.join(PARAMS['OUTPUT_PATH'], 'test_indices.csv')
    with open(test_indices_filename, 'rt') as f:
        test_indices = np.array([int(index) for index in f.readlines()])
    print('Loaded DZ indices')
    augmenter = augment.Augmenter(PARAMS)
    cur_kfold_sequences = augmenter.get_sequences(cross_val_indices)
    print('Created DZ sequences')
    cur_test_sequence = augmenter.get_test_sequence(test_indices)
    elp_train_indices_filename_pattern = os.path.join(
        _DEFAULT_PARAMS['OUTPUT_PATH'], 'elp_train_indices_%s.csv')
    elp_val_indices_filename_pattern = os.path.join(
        _DEFAULT_PARAMS['OUTPUT_PATH'], 'elp_val_indices_%s.csv')
    elp_cross_val_indices = []
    for i in range(5):
        with open(elp_train_indices_filename_pattern % i, 'rt') as f:
            elp_train_indices = np.array([int(index) for index in f.readlines()])
        with open(elp_val_indices_filename_pattern % i, 'rt') as f:
            elp_val_indices = np.array([int(index) for index in f.readlines()])
        elp_cross_val_indices.append((elp_train_indices, elp_val_indices))
    print('Loaded ELP indices')

    # Test encoder function that just flattens each spectrogram.
    encoder_func = lambda x: x.reshape(len(x), -1)[:_DEFAULT_PARAMS['REPRESENTATION_SIZE']]
    dz_triplet_generator = CreateTripletGenerator('JointTraining', cur_kfold_sequences[0].train, PARAMS, encoder_func)
    print('Created DZ generator')
    cur_elp_kfold_sequences = augmenter.get_elp_sequences(elp_cross_val_indices)
    print('Created ELP sequences')
    elp_triplet_generator = CreateTripletGenerator('JointTraining', cur_elp_kfold_sequences[0].train, PARAMS, encoder_func)
    print('Created ELP generator')
    triplet_generator = MultiDatasetTripletGenerator([dz_triplet_generator, elp_triplet_generator], PARAMS)
    print(len(triplet_generator))
    print(triplet_generator._index_to_generator_index)
    print(list(map(lambda x: (x[0].shape, x[1].shape, x[2].shape), triplet_generator[0])))
    for i in range(len(triplet_generator)):
        print(i)
        triplet_generator[i % len(triplet_generator)]
    print(list(map(lambda x: (x[0].shape, x[1].shape, x[2].shape), triplet_generator[0])))
    cur_elp_kfold_sequences[0].train.delete_data()
