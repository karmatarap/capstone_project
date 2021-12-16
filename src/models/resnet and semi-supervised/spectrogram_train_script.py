import collections
import contextlib
import copy
import gc
import json
import math
import os
import sys

import hyperopt
from hyperopt import hp
import matplotlib.pyplot as plt
import neptune.new as neptune
import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.utils
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa

from utils import common
from utils import cross_validation
from utils import lr_finder
from utils import sequence_utils
from utils import test_split


PARAM_CHOICE_VALUES = {
    'CONV_PADDING_TYPE': ['same', 'valid'],
    'POOLING_PADDING_TYPE': ['same', 'valid'],
    'ACTIVATION': ['relu', 'elu'],
    'OPTIMIZER': ['Adam', 'Nadam', 'RMSprop'],
    'POOLING_TYPE': ['avg', 'max'],
}

OPTIMIZER_NAME_TO_CLASS = {
    'Adam': tf.keras.optimizers.Adam,
    'Nadam': tf.keras.optimizers.Nadam,
    'RMSprop': tf.keras.optimizers.RMSprop,
}

POOLING_NAME_TO_CLASS = {
    'avg': tf.keras.layers.AveragePooling2D,
    'max': tf.keras.layers.MaxPooling2D,
}


def get_optimizer_class(cur_params):
    return OPTIMIZER_NAME_TO_CLASS[cur_params['OPTIMIZER']]


def get_pooling_class(cur_params):
    return POOLING_NAME_TO_CLASS[cur_params['POOLING_TYPE']]


def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    import random
    random.seed(seed)
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'
    np.random.seed(seed)
    tf.random.set_seed(seed)


@contextlib.contextmanager
def log_neptune():
    run = neptune.init(
        project=os.getenv('NEPTUNE_PROJECT'), api_token=os.getenv('NEPTUNE_API_TOKEN')
    )
    try:
        yield run
    finally:
        run.stop()


def log_static_value(neptune_logger, key, value, fold_index=None):
    prefix = ''
    if fold_index is not None:
        prefix = f'fold_{fold_index}/'
    neptune_logger[f'{prefix}{key}'] = value


def log_changing_value(neptune_logger, key, value, fold_index=None, iteration=None, epoch=None):
    prefix = ''
    if fold_index is not None:
        prefix += f'fold_{fold_index}/'
    # E.g. log training loss for semisupervised under fold_0/iteration_0/training_loss.
    # For non-semisupervised training loss, log under fold_0/training_loss.
    # For semi-supervised metrics that are not per epoch (e.g. num_new_ad_sa), log under fold_0/num_new_ad_sa.
    if iteration is not None and epoch is not None:
        prefix += f'iteration_{iteration}/'
    neptune_logger[f'{prefix}{key}'].log(value)


def log_results(neptune_logger, fold_index, prefix, results):
    log_static_value(neptune_logger, f'{prefix}/accuracy', results[0], fold_index=fold_index)
    # auc is set to -1 for per class results, since otherwise it would fail to compute.
    if results[1] >= 0:
        log_static_value(neptune_logger, f'{prefix}/auc', results[1], fold_index=fold_index)
    log_static_value(neptune_logger, f'{prefix}/precision', results[2], fold_index=fold_index)
    log_static_value(neptune_logger, f'{prefix}/recall', results[3], fold_index=fold_index)
    log_static_value(neptune_logger, f'{prefix}/cohen_kappa', results[4], fold_index=fold_index)
    log_static_value(neptune_logger, f'{prefix}/f1_macro', results[5], fold_index=fold_index)
    log_static_value(neptune_logger, f'{prefix}/f1_micro', results[6], fold_index=fold_index)


def log_history(neptune_logger, fold_index, iteration, is_pretrain_str, history):
    for key, values in history.history.items():
        for epoch, value in enumerate(values):
            log_changing_value(neptune_logger, f'{is_pretrain_str}/{key}', value, fold_index=fold_index, iteration=iteration, epoch=epoch)


def create_model_parts(cur_params):
    num_classes = cur_params['NUM_CLASSES']
    image_size = cur_params['IMAGE_SIZE']
    pretrain_last_conv_num_layers = cur_params['PRETRAIN_LAST_CONV_NUM_LAYERS']

    new_input = tf.keras.layers.Input(shape=(image_size, image_size, 3))
    model_to_function = {
        'ResNet50': tf.keras.applications.resnet.ResNet50,
        'ResNet101': tf.keras.applications.resnet.ResNet101,
        'ResNet152': tf.keras.applications.resnet.ResNet152,
    }
    pretrained_model = model_to_function[cur_params['MODEL']](
        include_top=False, 
        weights='imagenet',
        input_tensor=new_input)
    pretrained_model.trainable = False

    model = tf.keras.models.Sequential()
    model.add(pretrained_model)
    activation = cur_params['ACTIVATION']

    # Conv layers
    conv_kernel_size = cur_params['CONV_KERNEL_SIZE']
    pool_size = cur_params['POOL_SIZE']
    cur_conv_num_filters = cur_params['INITIAL_CONV_FILTERS']
    conv_multiplier = cur_params['CONV_FILTER_MULTIPLIER']
    for cur_conv_index in range(cur_params['NUM_CONV_LAYERS']):
        # Use ceil (here and below) to ensure it is always at least 1.
        model.add(tf.keras.layers.Conv2D(int(math.ceil(cur_conv_num_filters)), (conv_kernel_size, conv_kernel_size), padding=cur_params['CONV_PADDING_TYPE'], activation=activation))
        cur_conv_num_filters *= conv_multiplier
        if cur_conv_index < cur_params['NUM_POOLING_LAYERS']:
            model.add(get_pooling_class(cur_params)(pool_size=(pool_size, pool_size), padding=cur_params['POOLING_PADDING_TYPE']))
        if cur_conv_index < cur_params['NUM_CONV_DROPOUT_LAYERS']:
            model.add(tf.keras.layers.Dropout(cur_params['DROPOUT_RATE']))
        if cur_conv_index < cur_params['NUM_CONV_BATCH_NORM_LAYERS']:
            model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Flatten())

    # Dense layers
    cur_dense_size = cur_params['INITIAL_DENSE_SIZE']
    dense_multiplier = cur_params['DENSE_SIZE_MULTIPLIER']
    max_index = max(cur_params['NUM_DENSE_LAYERS'], cur_params['NUM_DENSE_DROPOUT_LAYERS'], cur_params['NUM_DENSE_BATCH_NORM_LAYERS'])
    for cur_dense_index in range(max_index):
        if cur_dense_index < cur_params['NUM_DENSE_LAYERS']:
            model.add(tf.keras.layers.Dense(int(math.ceil(cur_dense_size)), activation=activation))
        cur_dense_size *= dense_multiplier
        if cur_dense_index < cur_params['NUM_DENSE_DROPOUT_LAYERS']:
            model.add(tf.keras.layers.Dropout(cur_params['DROPOUT_RATE']))
        if cur_dense_index < cur_params['NUM_DENSE_BATCH_NORM_LAYERS']:
            model.add(tf.keras.layers.BatchNormalization())

    # soft-max layer
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    model.summary()
    
    # Split the model into a part that is frozen for all training and a part that is trainable.
    # The part that is frozen is used to precompute the parts of the data so that training can be faster and use less RAM and GPU memory.
    # The pretrained part that is trainable is frozen for the pretrain step to allow the classifier to train a little first.
    
    frozen_model = tf.keras.models.Model(pretrained_model.input,
                                         pretrained_model.layers[-pretrain_last_conv_num_layers - 1].output)
    non_frozen_input_shape = frozen_model.output_shape[1:]
    non_frozen_input = tf.keras.layers.Input(shape=non_frozen_input_shape)
    non_frozen_output = non_frozen_input

    for layer in pretrained_model.layers[-pretrain_last_conv_num_layers:-2]:
        non_frozen_output = layer(non_frozen_output)

    non_frozen_output = tf.keras.layers.Add()([non_frozen_input, non_frozen_output])
    non_frozen_output = pretrained_model.layers[-1](non_frozen_output)
    for layer in model.layers[1:]:
        non_frozen_output = layer(non_frozen_output)
    non_frozen_model = tf.keras.models.Model(non_frozen_input, non_frozen_output)
    non_frozen_model.compile(
        loss='categorical_crossentropy',
        optimizer=get_optimizer_class(cur_params)(lr=2e-5),
        metrics=['accuracy', 'AUC', 'Precision', 'Recall', tfa.metrics.CohenKappa(num_classes=num_classes),
                 tfa.metrics.F1Score(num_classes=num_classes, average='macro', name='f1_macro'),
                 tfa.metrics.F1Score(num_classes=num_classes, average='micro', name='f1_micro')])
    model.compile(
        loss='categorical_crossentropy',
        optimizer=get_optimizer_class(cur_params)(lr=2e-5),
        metrics=['accuracy', 'AUC', 'Precision', 'Recall', tfa.metrics.CohenKappa(num_classes=num_classes),
                 tfa.metrics.F1Score(num_classes=num_classes, average='macro', name='f1_macro'),
                 tfa.metrics.F1Score(num_classes=num_classes, average='micro', name='f1_micro')])
    return model, frozen_model, non_frozen_model


def train(cur_model, epochs, lr_epochs, min_lr, max_lr, train_sequence, val_sequence, es_patience, class_weights, cur_params, fold_index, neptune_logger, is_pretrain, iteration=None):
    cur_lr_finder = lr_finder.LRFinder(cur_model)
    cur_lr_finder.find_generator(train_sequence, min_lr, max_lr, epochs=lr_epochs, steps_per_epoch=len(train_sequence), verbose=2)
    save_path = os.path.join('backups', cur_params.get('HYPERPARAM_DIRNAME', 'hyperparameter_tuning'), 'lr_plots', f'{cur_params["NAME"]}_{cur_params["SEED"]}_{fold_index}_{"pretrain" if is_pretrain else "finetune"}.png')
    cur_lr_finder.plot_loss(save_path=save_path)
    try:
        best_lr = cur_lr_finder.get_best_lr(sma=3)
    except Exception:
        # Default to something reasonable (average of min and max lr) if best_lr fails.
        best_lr = (min_lr * max_lr) ** 0.5
    cur_lr_finder.delete_data()
    del cur_lr_finder
    print('Using LR:', best_lr)
    cur_model.compile(
        loss='categorical_crossentropy',
        optimizer=get_optimizer_class(cur_params)(lr=best_lr),
        metrics=['accuracy', 'AUC', 'Precision', 'Recall', tfa.metrics.CohenKappa(num_classes=cur_params['NUM_CLASSES']),
                 tfa.metrics.F1Score(num_classes=cur_params['NUM_CLASSES'], average='macro', name='f1_macro'),
                 tfa.metrics.F1Score(num_classes=cur_params['NUM_CLASSES'], average='micro', name='f1_micro')])
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=es_patience, restore_best_weights=True)]
    
    history = cur_model.fit(
        train_sequence,
        epochs=epochs,
        steps_per_epoch=len(train_sequence),
        verbose=2,
        validation_data=val_sequence,
        validation_steps=len(val_sequence),
        callbacks=callbacks,
        class_weight=class_weights
    )
    is_pretrain_str = 'pretrain' if is_pretrain else 'finetune'
    log_changing_value(neptune_logger, f'{is_pretrain_str}/best_lr', best_lr, fold_index=fold_index, iteration=iteration)
    log_history(neptune_logger, fold_index, iteration, is_pretrain_str, history)
    return history, best_lr


def predict_in_batch(cur_model, data, batch_size):
    predictions = []
    num_batches = int(math.ceil(len(data) / batch_size))
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = batch_start + batch_size
        predictions.extend(cur_model(data[batch_start:batch_end], training=False))
    return np.array(predictions)


def train_classification_model_semisupervised(
    epochs, lr_epochs, min_lr, train_sequence, val_sequence, es_patience, class_weights, cur_params, fold_index, neptune_logger):
    total_ad_sa = 0
    total_inf_juv = 0
    history = None
    best_lr = None
    cur_model, _, cur_non_frozen_model = create_model_parts(cur_params)
    cur_non_frozen_model.summary()
    classification_model = cur_model if cur_params['AUGMENTATION_ARGS'] else cur_non_frozen_model
    pretrain_history_and_best_lr = train(
        classification_model, cur_params['MAX_PRETRAIN_EPOCHS'], cur_params['PRETRAIN_LR_EPOCHS'], cur_params['MIN_PRETRAIN_LR'], cur_params['MAX_PRETRAIN_LR'],
        train_sequence, val_sequence, int(math.ceil(cur_params['PRETRAIN_ES_PATIENCE'])), class_weights, cur_params, fold_index, neptune_logger, is_pretrain=True)
    max_finetune_lr = pretrain_history_and_best_lr[1]
    if max_finetune_lr is None or max_finetune_lr > cur_params['MAX_FINETUNE_LR']:
        max_finetune_lr = cur_params['MAX_FINETUNE_LR']
    cur_non_frozen_model.trainable = True
    cur_non_frozen_model.summary()
    cur_model.summary()
    pretrain_weights = copy.deepcopy(classification_model.get_weights())
    for iteration in range(cur_params['MAX_SEMISUPERVISED_ITERATIONS']):
        if cur_params['RESET_TO_PRETRAIN_WEIGHTS']:
            classification_model.set_weights(copy.deepcopy(pretrain_weights))
        history, best_lr = train(
            classification_model, epochs, lr_epochs, min_lr, max_finetune_lr, train_sequence, val_sequence, es_patience,
            class_weights, cur_params, fold_index, neptune_logger, is_pretrain=False, iteration=iteration)
        if len(train_sequence.get_unlabeled_representations()) == 0:
            print(f'Added {total_ad_sa + total_inf_juv} labels ({total_ad_sa} ad/sa, {total_inf_juv} inf/juv)')
            return pretrain_history_and_best_lr, (history, best_lr), cur_non_frozen_model, cur_model
        cur_iteration_all_elp_predictions = predict_in_batch(classification_model, train_sequence.get_elp_representations(), cur_params['BATCH_SIZE'])
        unlabeled_predictions = predict_in_batch(classification_model, train_sequence.get_unlabeled_representations(), cur_params['BATCH_SIZE'])
        max_predictions = np.max(unlabeled_predictions, axis=1)
        valid_indices = np.where(max_predictions >= cur_params['MIN_THRESHOLD'])[0]
        print(len(unlabeled_predictions), len(valid_indices), np.sort(max_predictions)[:10])
        if not len(valid_indices):
            print(f'Added {total_ad_sa + total_inf_juv} labels ({total_ad_sa} ad/sa, {total_inf_juv} inf/juv)')
            return pretrain_history_and_best_lr, (history, best_lr), cur_non_frozen_model, cur_model
        unlabeled_predictions = unlabeled_predictions[valid_indices]
        max_predictions = np.max(unlabeled_predictions, axis=1)
        print(len(unlabeled_predictions), np.sort(max_predictions)[:10])
        sorted_prediction_indices = np.argsort(max_predictions)
        confident_prediction_indices = np.where(max_predictions >= cur_params['PREDICTION_THRESHOLD_FOR_NEW_LABEL'])[0]
        if len(confident_prediction_indices) < cur_params['MIN_ELP_TO_LABEL_PER_ITERATION']:

            confident_prediction_indices = sorted_prediction_indices[-cur_params['MIN_ELP_TO_LABEL_PER_ITERATION']:]
            print('Confidence not above threshold, adding highest with confidence',
                  unlabeled_predictions[confident_prediction_indices])
        confident_prediction_labels = np.argmax(unlabeled_predictions, axis=1)[confident_prediction_indices]
        train_sequence.add_elp_labels(confident_prediction_indices, confident_prediction_labels)
        num_inf_juv = np.sum(confident_prediction_labels)
        num_ad_sa = len(confident_prediction_indices) - num_inf_juv
        total_ad_sa += num_ad_sa
        total_inf_juv += num_inf_juv
        log_changing_value(neptune_logger, 'num_new_ad_sa', num_ad_sa, fold_index=fold_index, iteration=iteration)
        log_changing_value(neptune_logger, 'num_new_inf_juv', num_inf_juv, fold_index=fold_index, iteration=iteration)
        print(f'Added {len(confident_prediction_indices)} labels ({num_ad_sa} ad/sa, {num_inf_juv} inf/juv)')
        save_path = os.path.join('backups', cur_params.get('HYPERPARAM_DIRNAME', 'hyperparameter_tuning'), 'elp_predictions', f'{cur_params["NAME"]}_{cur_params["SEED"]}_{fold_index}_{iteration}_elp_predictions.npz')
        np.savez_compressed(save_path, all_elp=np.array(cur_iteration_all_elp_predictions), added=np.array(unlabeled_predictions[confident_prediction_indices]))
        save_predictions(cur_model, cur_params, fold_index, iteration=iteration)
        del cur_iteration_all_elp_predictions, unlabeled_predictions, max_predictions, sorted_prediction_indices, confident_prediction_indices, confident_prediction_labels
        gc.collect()
    print(f'Added {total_ad_sa + total_inf_juv} labels ({total_ad_sa} ad/sa, {total_inf_juv} inf/juv)')
    return pretrain_history_and_best_lr, (history, best_lr), cur_non_frozen_model, cur_model


def plot_accuracy_and_loss(pretrain_history, finetune_history, cur_params, fold_index):
    acc = pretrain_history.history['accuracy'] + finetune_history.history['accuracy']
    val_acc = pretrain_history.history['val_accuracy'] + finetune_history.history['val_accuracy']
    loss = pretrain_history.history['loss'] + finetune_history.history['loss']
    val_loss = pretrain_history.history['val_loss'] + finetune_history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'bo', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(os.path.join('backups', cur_params.get('HYPERPARAM_DIRNAME', 'hyperparameter_tuning'), 'history_plots', f'{cur_params["NAME"]}_{cur_params["SEED"]}_{fold_index}_accuracy.png'))
    plt.close()

    plt.figure()

    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'ro', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.savefig(os.path.join('backups', cur_params.get('HYPERPARAM_DIRNAME', 'hyperparameter_tuning'), 'history_plots', f'{cur_params["NAME"]}_{cur_params["SEED"]}_{fold_index}_loss.png'))
    plt.close()


def get_value_for_metric(metric, y_true, y_pred):
    metric.update_state(y_true, y_pred)
    return metric.result().numpy()


def evaluate_with_threshold(cur_model, sequence, ad_sa_threshold, cur_params, ignore_auc=False):
    results_without_threshold = cur_model.evaluate(sequence, verbose=0)
    loss = results_without_threshold[0]
    results_without_threshold = results_without_threshold[1:]
    if ad_sa_threshold is None:
        return results_without_threshold, results_without_threshold, loss
    y_pred = cur_model.predict(sequence)
    y_true = sequence.classes
    cur_y_pred = np.zeros((len(y_true),))
    cur_y_pred[y_pred[:, 0] < ad_sa_threshold] = 1
    cur_y_pred = cur_y_pred.tolist()
    accuracy = sklearn.metrics.accuracy_score(y_true, cur_y_pred, normalize=True)
    if ignore_auc:
        auc = -1
    else:
        auc = sklearn.metrics.roc_auc_score(y_true, cur_y_pred, average='micro')

    precision = sklearn.metrics.precision_score(y_true, cur_y_pred, average='micro')
    recall = sklearn.metrics.recall_score(y_true, cur_y_pred, average='micro')
    cohen_kappa = get_value_for_metric(tfa.metrics.CohenKappa(num_classes=cur_params['NUM_CLASSES']), y_true, cur_y_pred)
    f1_macro = sklearn.metrics.f1_score(y_true, cur_y_pred, average='macro')
    f1_micro = sklearn.metrics.f1_score(y_true, cur_y_pred, average='micro')
    results_with_threshold = np.array([accuracy, auc, precision, recall, cohen_kappa, f1_macro, f1_micro])
    return results_without_threshold, results_with_threshold, loss


def tune_threshold(cur_model, sequence):
    y_pred = cur_model.predict(sequence)
    y_true = sequence.classes
    def objective(ad_sa_threshold):
        cur_y_pred = np.zeros((len(y_true),))
        cur_y_pred[y_pred[:, 0] < ad_sa_threshold] = 1
        return -sklearn.metrics.f1_score(y_true, cur_y_pred.tolist(), average='macro')
    best_threshold = hyperopt.fmin(fn=objective,
        space=hp.uniform('ad_sa_threshold', 0.0, 1.0),
        algo=hyperopt.tpe.suggest,
        max_evals=100)['ad_sa_threshold']
    print('Best threshold is', best_threshold)
    return best_threshold


def plot_confusion_matrix(cur_model, sequence, cur_params, fold_index, ad_sa_threshold):
    y_pred = cur_model.predict(sequence)
    y_true = sequence.classes
    if ad_sa_threshold is None:
        y_pred = np.argmax(y_pred, axis=1)
        threshold_str = 'before_thresholding'
    else:
        cur_y_pred = np.zeros((len(y_true),), dtype=int)
        cur_y_pred[y_pred[:, 0] < ad_sa_threshold] = 1
        y_pred = cur_y_pred
        threshold_str = 'after_thresholding'
    class_labels = list(sequence.class_indices.keys())
    y_pred = list(map(lambda x: class_labels[x], y_pred))
    y_true = list(map(lambda x: class_labels[x], y_true))

    cm_unnormalized = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=class_labels)
    cm_precision = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=class_labels, normalize='pred')
    cm_recall = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=class_labels, normalize='true')
    disp_unnormalized = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm_unnormalized, display_labels=class_labels)
    # print('Class counts')
    disp_unnormalized.plot()
    plt.savefig(os.path.join('backups', cur_params.get('HYPERPARAM_DIRNAME', 'hyperparameter_tuning'), 'confusion_matrices', f'{cur_params["NAME"]}_{cur_params["SEED"]}_{fold_index}_{threshold_str}_class_counts.png'))
    plt.close()
    disp_precision = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm_precision, display_labels=class_labels)
    # print('Precision')
    disp_precision.plot()
    plt.savefig(os.path.join('backups', cur_params.get('HYPERPARAM_DIRNAME', 'hyperparameter_tuning'), 'confusion_matrices', f'{cur_params["NAME"]}_{cur_params["SEED"]}_{fold_index}_{threshold_str}_precision.png'))
    plt.close()
    disp_recall = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm_recall, display_labels=class_labels)
    # print('Recall')
    disp_recall.plot()
    plt.savefig(os.path.join('backups', cur_params.get('HYPERPARAM_DIRNAME', 'hyperparameter_tuning'), 'confusion_matrices', f'{cur_params["NAME"]}_{cur_params["SEED"]}_{fold_index}_{threshold_str}_recall.png'))
    plt.close()


def get_class_accuracies(frozen_model, cur_models, seed, possible_labels, ad_sa_thresholds, params, neptune_logger):
    cur_params = params.copy()
    cur_params['SEED'] = seed
    print('Class accuracies')
    # Note that train class accuracies do not include any ELP rumbles given a label by semi-supervised.
    print('Train')
    for i in range(cur_params['START_FOLD'], cur_params['START_FOLD'] + cur_params['NUM_FOLDS_TO_TRAIN']):
        train_sequence = sequence_utils.SpectrogramClassificationSequenceWithLabels(
            i, sequence_utils.TRAIN_FOLD_TYPE, frozen_model, possible_labels, include_elp=False, params=cur_params)
        ad_sa_train_sequence = sequence_utils.FilteredClassificationSequenceWithLabels(
            train_sequence, 'ad/sa', params=cur_params)
        inf_juv_train_sequence = sequence_utils.FilteredClassificationSequenceWithLabels(
            train_sequence, 'inf/juv', params=cur_params)
        print('')
        print('FOLD', i)
        print('Train ad/sa before and after thresholding')
        train_results_ad_sa = evaluate_with_threshold(cur_models[i - cur_params['START_FOLD']], ad_sa_train_sequence, ad_sa_thresholds[i - cur_params['START_FOLD']], cur_params, ignore_auc=True)
        print(train_results_ad_sa)
        print('Train inf/juv before and after thresholding')
        train_results_inf_juv = evaluate_with_threshold(cur_models[i - cur_params['START_FOLD']], inf_juv_train_sequence, ad_sa_thresholds[i - cur_params['START_FOLD']], cur_params, ignore_auc=True)
        print(train_results_inf_juv)
        log_results(neptune_logger, i, 'train/ad_sa/before_thresholding', train_results_ad_sa[0])
        log_results(neptune_logger, i, 'train/ad_sa/after_thresholding', train_results_ad_sa[1])
        log_results(neptune_logger, i, 'train/inf_juv/before_thresholding', train_results_inf_juv[0])
        log_results(neptune_logger, i, 'train/inf_juv/after_thresholding', train_results_inf_juv[1])
        train_sequence.delete_data()
        ad_sa_train_sequence.delete_data()
        inf_juv_train_sequence.delete_data()
    print('')
    print('Val')
    val_class_accuracies = []
    val_class_accuracies_before_thresholding = []
    for i in range(cur_params['START_FOLD'], cur_params['START_FOLD'] + cur_params['NUM_FOLDS_TO_TRAIN']):
        val_sequence = sequence_utils.SpectrogramClassificationSequenceWithLabels(
            i, sequence_utils.VAL_FOLD_TYPE, frozen_model, possible_labels, include_elp=False, params=cur_params)
        ad_sa_val_sequence = sequence_utils.FilteredClassificationSequenceWithLabels(
            val_sequence, 'ad/sa', params=cur_params)
        inf_juv_val_sequence = sequence_utils.FilteredClassificationSequenceWithLabels(
            val_sequence, 'inf/juv', params=cur_params)
        print('')
        print('FOLD', i)
        print('Val ad/sa before and after thresholding')
        val_results_ad_sa = evaluate_with_threshold(cur_models[i - cur_params['START_FOLD']], ad_sa_val_sequence, ad_sa_thresholds[i - cur_params['START_FOLD']], cur_params, ignore_auc=True)
        print(val_results_ad_sa)
        print('Val inf/juv before and after thresholding')
        val_results_inf_juv = evaluate_with_threshold(cur_models[i - cur_params['START_FOLD']], inf_juv_val_sequence, ad_sa_thresholds[i - cur_params['START_FOLD']], cur_params, ignore_auc=True)
        print(val_results_inf_juv)
        val_class_accuracies_before_thresholding.append((val_results_ad_sa[0][0], val_results_inf_juv[0][0]))
        val_class_accuracies.append((val_results_ad_sa[1][0], val_results_inf_juv[1][0]))
        log_results(neptune_logger, i, 'val/ad_sa/before_thresholding', val_results_ad_sa[0])
        log_results(neptune_logger, i, 'val/ad_sa/after_thresholding', val_results_ad_sa[1])
        log_results(neptune_logger, i, 'val/inf_juv/before_thresholding', val_results_inf_juv[0])
        log_results(neptune_logger, i, 'val/inf_juv/after_thresholding', val_results_inf_juv[1])
        val_sequence.delete_data()
        ad_sa_val_sequence.delete_data()
        inf_juv_val_sequence.delete_data()
    test_class_accuracies = []
    test_class_accuracies_before_thresholding = []
    print('')
    print('Test')
    for i in range(cur_params['START_FOLD'], cur_params['START_FOLD'] + cur_params['NUM_FOLDS_TO_TRAIN']):
        test_sequence = sequence_utils.SpectrogramClassificationTestSequence(
            frozen_model, possible_labels, params=cur_params)
        ad_sa_test_sequence = sequence_utils.FilteredClassificationSequenceWithLabels(
            test_sequence, 'ad/sa', params=cur_params)
        inf_juv_test_sequence = sequence_utils.FilteredClassificationSequenceWithLabels(
            test_sequence, 'inf/juv', params=cur_params)
        print('')
        print('FOLD', i)
        print('Test ad/sa before and after thresholding')
        test_results_ad_sa = evaluate_with_threshold(cur_models[i - cur_params['START_FOLD']], ad_sa_test_sequence, ad_sa_thresholds[i - cur_params['START_FOLD']], cur_params, ignore_auc=True)
        print(test_results_ad_sa)
        print('Test inf/juv before and after thresholding')
        test_results_inf_juv = evaluate_with_threshold(cur_models[i - cur_params['START_FOLD']], inf_juv_test_sequence, ad_sa_thresholds[i - cur_params['START_FOLD']], cur_params, ignore_auc=True)
        print(test_results_inf_juv)
        test_class_accuracies_before_thresholding.append((test_results_ad_sa[0][0], test_results_inf_juv[0][0]))
        test_class_accuracies.append((test_results_ad_sa[1][0], test_results_inf_juv[1][0]))
        log_results(neptune_logger, i, 'test/ad_sa/before_thresholding', test_results_ad_sa[0])
        log_results(neptune_logger, i, 'test/ad_sa/after_thresholding', test_results_ad_sa[1])
        log_results(neptune_logger, i, 'test/inf_juv/before_thresholding', test_results_inf_juv[0])
        log_results(neptune_logger, i, 'test/inf_juv/after_thresholding', test_results_inf_juv[1])
        test_sequence.delete_data()
        ad_sa_test_sequence.delete_data()
        inf_juv_test_sequence.delete_data()
    return val_class_accuracies_before_thresholding, val_class_accuracies, test_class_accuracies_before_thresholding, test_class_accuracies


def get_data_from_sequence(sequence):
    x_data, y_data = [], []
    for i in range(len(sequence)):
        cur_x, cur_y = sequence[i]
        x_data.extend(cur_x)
        y_data.extend(cur_y)
    return np.array(x_data), np.array(y_data)


def save_predictions(cur_model, cur_params, fold_index, iteration=None):
    dz_df = common.load_dz_data(cur_params['BASE_DATA_DIR'])
    spectrograms = common.load_dz_spectrogram_data(dz_df, cur_params['IMAGE_SIZE'], cur_params['SEED'], preprocessing=common.get_preprocessing_func(cur_params['MODEL']))
    predictions = predict_in_batch(cur_model, spectrograms, cur_params['BATCH_SIZE'])
    iteration_str = f'_{iteration}' if iteration is not None else ''
    save_path = os.path.join('backups', cur_params.get('HYPERPARAM_DIRNAME', 'hyperparameter_tuning'), 'dz_predictions', f'{cur_params["NAME"]}_{cur_params["SEED"]}_{fold_index}{iteration_str}_dz_predictions.npz')
    np.savez_compressed(save_path, dz_predictions=predictions)


def save_data(cross_val_fold_sequences, test_sequence, numpy_path, cur_params):
    numpy_data = {}
    for i in range(cur_params['NUM_K_FOLDS']):
        val_x, val_y = get_data_from_sequence(cross_val_fold_sequences[i].val)
        numpy_data[f'val_x_{i}'] = val_x
        numpy_data[f'val_y_{i}'] = val_y
    test_x, test_y = get_data_from_sequence(test_sequence)
    numpy_data['test_x'] = test_x
    numpy_data['test_y'] = test_y
    np.savez(numpy_path, **numpy_data)
    

def save_model(model_to_save, model_path):
    model_to_save.save(model_path)


def run_everything_for_seed(params, neptune_logger):
    # Releases memory and removes old models.
    K.clear_session()
    cur_params = params.copy()
    seed = cur_params['SEED']
    set_seeds(seed)
    cur_params['TRAIN_TEST_SPLIT_SEED'] = seed
    neptune_logger['params'] = cur_params.copy()
    cur_non_frozen_models = []
    ad_sa_thresholds = []
    val_results_before_thresholding = []
    val_results_after_thresholding = []
    val_losses = []
    test_results_before_thresholding = []
    test_results_after_thresholding = []
    histories_and_best_lrs = []
    df = common.load_dz_data(cur_params['BASE_DATA_DIR'])
    possible_labels = sorted(list(df[cur_params['TARGET_COL']].unique()))
    test_split.TestSplitter(cur_params).get_no_leakage_trainval_test_splits()
    
    train_val_indices_filename = os.path.join(cur_params['OUTPUT_PATH'], 'train_val_indices.csv')
    with open(train_val_indices_filename, 'rt') as f:
        train_val_indices = np.array([int(index) for index in f.readlines()])
    cross_validator = cross_validation.CrossValidator(cur_params)
    cross_val_indices = cross_validator.get_no_leakage_crossval_splits(train_val_indices)
    
    train_val_df = df.iloc[train_val_indices]
    if cur_params.get('UPSAMPLE_TO_LARGEST_CLASS', False):
        class_weights = {i: 1.0 for i in range(len(possible_labels))}
    else:
        one_hot_labels = pd.get_dummies(train_val_df[cur_params['TARGET_COL']].astype(pd.CategoricalDtype(
            categories=possible_labels)))
        labels = np.argmax(one_hot_labels.to_numpy(), axis=1)
        unique_labels = np.unique(labels)
        class_weights = dict(zip(unique_labels, sklearn.utils.class_weight.compute_class_weight(
            'balanced', classes=unique_labels, y=labels)))
    print('Class weights:', class_weights)
    _, frozen_model, _ = create_model_parts(cur_params)
    cur_test_sequence = sequence_utils.SpectrogramClassificationTestSequence(
        frozen_model, possible_labels, params=cur_params)
    for fold_index in range(cur_params['START_FOLD'], cur_params['START_FOLD'] + cur_params['NUM_FOLDS_TO_TRAIN']):
        print('')
        print('')
        print('')
        print('FOLD', fold_index)
        set_seeds(seed)
        classification_train_seq = sequence_utils.SpectrogramClassificationSequenceWithLabels(
            fold_index, sequence_utils.TRAIN_FOLD_TYPE, frozen_model=None if cur_params['AUGMENTATION_ARGS'] else frozen_model, possible_labels=possible_labels, include_elp=cur_params['SEMI_SUPERVISED'], params=cur_params)
        classification_val_seq = sequence_utils.SpectrogramClassificationSequenceWithLabels(
            fold_index, sequence_utils.VAL_FOLD_TYPE, frozen_model=None if cur_params['AUGMENTATION_ARGS'] else frozen_model, possible_labels=possible_labels, include_elp=False, params=cur_params)
        set_seeds(seed)
        if cur_params['SEMI_SUPERVISED']:
            pretrain_history_and_best_lr, finetune_history_and_best_lr, cur_non_frozen_model, cur_model = train_classification_model_semisupervised(
                cur_params['MAX_FINETUNE_EPOCHS'], cur_params['FINETUNE_LR_EPOCHS'], cur_params['MIN_FINETUNE_LR'], classification_train_seq, classification_val_seq, int(math.ceil(cur_params['FINETUNE_ES_PATIENCE'])), class_weights, cur_params, fold_index, neptune_logger)
        else:
            cur_model, _, cur_non_frozen_model = create_model_parts(cur_params)
            cur_non_frozen_model.summary()
            pretrain_history_and_best_lr = train(
                cur_model if cur_params['AUGMENTATION_ARGS'] else cur_non_frozen_model, cur_params['MAX_PRETRAIN_EPOCHS'], cur_params['PRETRAIN_LR_EPOCHS'], cur_params['MIN_PRETRAIN_LR'], cur_params['MAX_PRETRAIN_LR'],
                classification_train_seq, classification_val_seq, int(math.ceil(cur_params['PRETRAIN_ES_PATIENCE'])), class_weights, cur_params, fold_index, neptune_logger, is_pretrain=True)
            max_finetune_lr = pretrain_history_and_best_lr[1]
            if max_finetune_lr is None or max_finetune_lr > cur_params['MAX_FINETUNE_LR']:
                max_finetune_lr = cur_params['MAX_FINETUNE_LR']
            cur_non_frozen_model.trainable = True
            cur_non_frozen_model.summary()
            cur_model.summary()
            finetune_history_and_best_lr = train(
                cur_model if cur_params['AUGMENTATION_ARGS'] else cur_non_frozen_model, cur_params['MAX_FINETUNE_EPOCHS'], cur_params['FINETUNE_LR_EPOCHS'], cur_params['MIN_FINETUNE_LR'], max_finetune_lr, classification_train_seq, classification_val_seq, int(math.ceil(cur_params['FINETUNE_ES_PATIENCE'])), class_weights, cur_params, fold_index, neptune_logger, is_pretrain=False)
        
        
        classification_train_seq.delete_data()
        classification_val_seq.delete_data()
        histories_and_best_lrs.append((pretrain_history_and_best_lr,
                                       finetune_history_and_best_lr))

        plot_accuracy_and_loss(pretrain_history_and_best_lr[0],
                               finetune_history_and_best_lr[0],
                               cur_params,
                               fold_index)
        classification_val_seq_frozen = sequence_utils.SpectrogramClassificationSequenceWithLabels(
            fold_index, sequence_utils.VAL_FOLD_TYPE, frozen_model=frozen_model, possible_labels=possible_labels, include_elp=False, params=cur_params)
        if cur_params['NUM_CLASSES'] == 2:
            ad_sa_threshold = tune_threshold(cur_non_frozen_model, classification_val_seq_frozen)
        else:
            ad_sa_threshold = None
        log_static_value(neptune_logger, 'ad_sa_threshold', ad_sa_threshold, fold_index=fold_index)
        ad_sa_thresholds.append(ad_sa_threshold)
        val_result = evaluate_with_threshold(cur_non_frozen_model, classification_val_seq_frozen, ad_sa_threshold, cur_params)
        print('Val results before thresholding')
        print(val_result[0])
        print('Val results after thresholding')
        print(val_result[1])
        val_results_before_thresholding.append(val_result[0])
        val_results_after_thresholding.append(val_result[1])
        log_results(neptune_logger, fold_index, 'val/before_thresholding', val_result[0])
        log_results(neptune_logger, fold_index, 'val/after_thresholding', val_result[1])
        log_static_value(neptune_logger, 'val/loss', val_result[2], fold_index=fold_index)
        val_losses.append(val_result[2])
        classification_val_seq_frozen.delete_data()
        test_result = evaluate_with_threshold(cur_non_frozen_model, cur_test_sequence, ad_sa_threshold, cur_params)
        print('Test results before thresholding')
        print(test_result[0])
        print('Test results after thresholding')
        print(test_result[1])
        test_results_before_thresholding.append(test_result[0])
        test_results_after_thresholding.append(test_result[1])
        log_results(neptune_logger, fold_index, 'test/before_thresholding', test_result[0])
        log_results(neptune_logger, fold_index, 'test/after_thresholding', test_result[1])
        plot_confusion_matrix(cur_non_frozen_model, cur_test_sequence, cur_params, fold_index, ad_sa_threshold=None)
        if ad_sa_threshold is not None:
            plot_confusion_matrix(cur_non_frozen_model, cur_test_sequence, cur_params, fold_index, ad_sa_threshold=ad_sa_threshold)
        save_predictions(cur_model, cur_params, fold_index)
        model_save_path = f'backups/{cur_params.get("HYPERPARAM_DIRNAME", "hyperparameter_tuning")}/models/{cur_params["NAME"]}_{seed}_model_{fold_index}.h5'
        # save_model(cur_model, model_save_path)
        # neptune_logger[f'fold_{fold_index}/model_path'].track_files(model_save_path)
        cur_non_frozen_models.append(cur_non_frozen_model)
        del cur_model
    cur_test_sequence.delete_data()
    if cur_params['NUM_CLASSES'] == 2:
        class_accuracies = get_class_accuracies(frozen_model, cur_non_frozen_models, seed, possible_labels, ad_sa_thresholds, cur_params, neptune_logger)
    else:
        class_accuracies = None
    # Delete the models so Tensorflow frees the memory for more models.
    for model in cur_non_frozen_models:
        del model
    del cur_test_sequence
    # Garbage collect (free memory for) anything that was deleted and is still using memory.
    gc.collect()
    return ((val_results_before_thresholding, val_results_after_thresholding,test_results_before_thresholding, test_results_after_thresholding, class_accuracies, val_losses), cross_val_indices, ad_sa_thresholds, histories_and_best_lrs)


def construct_results_df(results, cur_params):
    results_dict = {
        # Random seed used for splitting the data, initializing the models, training, etc.
        'Seed': [],
        'Fold': [],
        'Val accuracy': [],
        'Val f1 macro': [],
        'Val accuracy before thresholding': [],
        'Val f1 macro before thresholding': [],
    }
    class_accuracies_results_dict = {
        'ad/sa val accuracy': [],
        'inf/juv val accuracy': [],
        'ad/sa val accuracy before thresholding': [],
        'inf/juv val accuracy before thresholding': [],
        'ad/sa test accuracy': [],
        'inf/juv test accuracy': [],
        'ad/sa test accuracy before thresholding': [],
        'inf/juv test accuracy before thresholding': [],
    }
    if cur_params['NUM_CLASSES'] == 2:
        results_dict.update(class_accuracies_results_dict)
    test_metrics_results_dict = {
        'Test accuracy': [],
        'Test f1 macro': [],
        'Test accuracy before thresholding': [],
        'Test f1 macro before thresholding': [],
    }
    results_dict.update(test_metrics_results_dict)
    results_dict['Val loss'] = []
    # Other columns just for saving the data
    if cur_params['NUM_CLASSES'] == 2:
        results_dict['ad_sa_threshold'] = []
    results_dict['history'] = []

    val_results_before_thresholding = results[0][0]
    val_results = results[0][1]
    test_results_before_thresholding = results[0][2]
    test_results = results[0][3]
    if cur_params['NUM_CLASSES'] == 2:
        val_class_accuracies_before_thresholding = results[0][4][0]
        val_class_accuracies = results[0][4][1]
        test_class_accuracies_before_thresholding = results[0][4][2]
        test_class_accuracies = results[0][4][3]
    for i in range(cur_params['NUM_FOLDS_TO_TRAIN']):
        results_dict['Seed'].append(cur_params['SEED'])
        results_dict['Fold'].append(i)
        results_dict['Val accuracy'].append(val_results[i][0])
        results_dict['Val f1 macro'].append(val_results[i][5])
        results_dict['Val accuracy before thresholding'].append(val_results_before_thresholding[i][0])
        results_dict['Val f1 macro before thresholding'].append(val_results_before_thresholding[i][5])
        results_dict['Test accuracy'].append(test_results[i][0])
        results_dict['Test f1 macro'].append(test_results[i][5])
        results_dict['Test accuracy before thresholding'].append(test_results_before_thresholding[i][0])
        results_dict['Test f1 macro before thresholding'].append(test_results_before_thresholding[i][5])
        if cur_params['NUM_CLASSES'] == 2:
            results_dict['ad/sa val accuracy'].append(val_class_accuracies[i][0])
            results_dict['inf/juv val accuracy'].append(val_class_accuracies[i][1])
            results_dict['ad/sa val accuracy before thresholding'].append(val_class_accuracies_before_thresholding[i][0])
            results_dict['inf/juv val accuracy before thresholding'].append(val_class_accuracies_before_thresholding[i][1])
            results_dict['ad/sa test accuracy'].append(test_class_accuracies[i][0])
            results_dict['inf/juv test accuracy'].append(test_class_accuracies[i][1])
            results_dict['ad/sa test accuracy before thresholding'].append(test_class_accuracies_before_thresholding[i][0])
            results_dict['inf/juv test accuracy before thresholding'].append(test_class_accuracies_before_thresholding[i][1])
        results_dict['Val loss'].append(results[0][5][i])
        if cur_params['NUM_CLASSES'] == 2:
            results_dict['ad_sa_threshold'].append(results[2][i])
        results_dict['history'].append(results[3][i][0])

    results_df = pd.DataFrame.from_dict(results_dict)
    return results_df


def plot_accuracies(results_df, cur_params):
    plt.scatter(results_df['inf/juv val accuracy'], results_df['inf/juv test accuracy'], label='inf/juv')
    plt.scatter(results_df['ad/sa val accuracy'], results_df['ad/sa test accuracy'], label='ad/sa')
    plt.scatter(results_df['Val accuracy'], results_df['Test accuracy'], label='overall')
    plt.plot(np.linspace(0, 1, 3), np.linspace(0, 1, 3), '--')
    plt.xlabel('Validation accuracy')
    plt.ylabel('Test accuracy')
    plt.title('Validation vs test accuracy')
    plt.legend()
    plt.savefig(os.path.join('backups', cur_params.get('HYPERPARAM_DIRNAME', 'hyperparameter_tuning'), 'test_accuracy_plots', f'{cur_params["NAME"]}_{cur_params["START_FOLD"]}.png'))
    plt.close()


def save_indices(results, cur_params):
    indices_dict = {}
    for i in range(cur_params['NUM_K_FOLDS']):
        indices_dict[f'seed_{cur_params["SEED"]}_val_{i}_indices'] = results[1][i][1]
    np.savez(f'backups/{cur_params.get("HYPERPARAM_DIRNAME", "hyperparameter_tuning")}/indices/{cur_params["NAME"]}_indices.npz', **indices_dict)


def main(argv):
    assert(len(argv) >= 1)
    params_json_path = argv[1]
    with open(params_json_path, 'rt') as f:
        params = json.load(f)
    all_results = {}
    with log_neptune() as neptune_logger:
        all_results = run_everything_for_seed(params, neptune_logger)
    results_df = construct_results_df(all_results, params)
    results_df.to_csv(os.path.join('backups', params.get('HYPERPARAM_DIRNAME', 'hyperparameter_tuning'), 'results', f'{params["NAME"]}_{params["START_FOLD"]}.csv'))
    if params['NUM_CLASSES'] == 2:
        plot_accuracies(results_df, params)
    save_indices(all_results, params)


if __name__ == '__main__':
    main(sys.argv)