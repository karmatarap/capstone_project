import json
import os
import pickle
import subprocess

import hyperopt
from hyperopt import hp
import numpy as np
import pandas as pd

from utils import common

import spectrogram_train_script

_BASE_HYPERPARAM_DIRNAME = 'hyperparameter_tuning'


CONSTANT_PARAMS = {
    'AUGMENTATION_ARGS': {},
    'BASE_DATA_DIR': 'dzanga-bai-20210816T230919Z-001/dzanga-bai',
    'BASE_ELP_SPECTROGRAM_DATA_DIR': 'elp_data/spectrograms',
    'IMAGE_SIZE': 512,
    'TARGET_COL': 'agecat',
    'NUM_K_FOLDS': 5,
    'STRATIFY_COL': 'age',
    'OUTPUT_PATH': 'dzanga-bai-20210816T230919Z-001/dzanga-bai',
    'NUM_CLASSES': 2,
    # Use this to reduce the folds for hyperparameter tuning.
    # Initially use 1 to explore more of the search space and then set to NUM_K_FOLDS to learn more about points in the good regions.
    'NUM_FOLDS_TO_TRAIN': 5,
    'START_FOLD': 0,
    'PRETRAIN_LAST_CONV_NUM_LAYERS': 10,
    'MODEL': 'ResNet152',
    'ALLOW_LEAKAGE': False,
    'SEMI_SUPERVISED': False,
    'VAL_SIZE': None,
}


def add_dependent_uniform_param(param_name, max_param_name, params):
    ratio_name = f'{param_name}_RATIO'
    params[ratio_name] = hp.uniform(ratio_name, 0, 1)
    params[param_name] = params[ratio_name] * params[max_param_name]


def create_searchable_params_space():
    searchable_params = {
        'DROPOUT_RATE': hp.uniform('DROPOUT_RATE', 0.1, 0.9),
        'BATCH_SIZE': hp.qloguniform('BATCH_SIZE', np.log(2), np.log(64), 1),
        'NUM_CONV_LAYERS': hp.quniform('NUM_CONV_LAYERS', 1, 4, 1),
        'NUM_DENSE_LAYERS': hp.quniform('NUM_DENSE_LAYERS', 1, 4, 1),
        'CONV_KERNEL_SIZE': hp.quniform('CONV_KERNEL_SIZE', 1, 7, 1),
        'POOL_SIZE': hp.quniform('POOL_SIZE', 1, 7, 1),
        'POOLING_TYPE': hp.choice('POOLING_TYPE', spectrogram_train_script.PARAM_CHOICE_VALUES['POOLING_TYPE']),
        'CONV_PADDING_TYPE': hp.choice('CONV_PADDING_TYPE', spectrogram_train_script.PARAM_CHOICE_VALUES['CONV_PADDING_TYPE']),
        'POOLING_PADDING_TYPE': hp.choice('POOLING_PADDING_TYPE', spectrogram_train_script.PARAM_CHOICE_VALUES['POOLING_PADDING_TYPE']),
        'INITIAL_CONV_FILTERS': hp.qloguniform('INITIAL_CONV_FILTERS', np.log(2), np.log(1024), 1),
        'INITIAL_DENSE_SIZE': hp.qloguniform('INITIAL_DENSE_SIZE', np.log(2), np.log(1024), 1),
        'CONV_FILTER_MULTIPLIER': hp.loguniform('CONV_FILTER_MULTIPLIER', np.log(1/16), np.log(16)),
        'DENSE_SIZE_MULTIPLIER': hp.loguniform('DENSE_SIZE_MULTIPLIER', np.log(1/16), np.log(16)),
        'ACTIVATION': hp.choice('ACTIVATION', spectrogram_train_script.PARAM_CHOICE_VALUES['ACTIVATION']),
        'OPTIMIZER': hp.choice('OPTIMIZER', spectrogram_train_script.PARAM_CHOICE_VALUES['OPTIMIZER']),
        'MAX_PRETRAIN_EPOCHS': hp.qloguniform('MAX_PRETRAIN_EPOCHS', np.log(10), np.log(100), 1),
        'MAX_FINETUNE_EPOCHS': hp.qloguniform('MAX_FINETUNE_EPOCHS', np.log(10), np.log(100), 1),
        'PRETRAIN_LR_EPOCHS': hp.quniform('PRETRAIN_LR_EPOCHS', 2, 10, 1),
        'FINETUNE_LR_EPOCHS': hp.quniform('FINETUNE_LR_EPOCHS', 2, 10, 1),
        'PRETRAIN_ES_PATIENCE_RATIO': hp.loguniform('PRETRAIN_ES_PATIENCE_RATIO', np.log(1/100), np.log(1)),
        'FINETUNE_ES_PATIENCE_RATIO': hp.loguniform('FINETUNE_ES_PATIENCE_RATIO', np.log(1/100), np.log(1)),
        'MIN_PRETRAIN_LR_RATIO': hp.loguniform('MIN_PRETRAIN_LR_RATIO', np.log(1e-4), np.log(1)),
        'MIN_FINETUNE_LR_RATIO': hp.loguniform('MIN_FINETUNE_LR_RATIO', np.log(1e-4), np.log(1)),
        'MAX_PRETRAIN_LR': hp.loguniform('MAX_PRETRAIN_LR', np.log(1e-6), np.log(1)),
        'MAX_FINETUNE_LR': hp.loguniform('MAX_FINETUNE_LR', np.log(1e-6), np.log(1)),
    }

    searchable_params['PRETRAIN_ES_PATIENCE'] = searchable_params['MAX_PRETRAIN_EPOCHS'] * searchable_params['PRETRAIN_ES_PATIENCE_RATIO']
    searchable_params['FINETUNE_ES_PATIENCE'] = searchable_params['MAX_FINETUNE_EPOCHS'] * searchable_params['FINETUNE_ES_PATIENCE_RATIO']

    searchable_params['MIN_PRETRAIN_LR'] = searchable_params['MAX_PRETRAIN_LR'] * searchable_params['MIN_PRETRAIN_LR_RATIO']
    searchable_params['MIN_FINETUNE_LR'] = searchable_params['MAX_FINETUNE_LR'] * searchable_params['MIN_FINETUNE_LR_RATIO']

    add_dependent_uniform_param('NUM_POOLING_LAYERS', 'NUM_CONV_LAYERS', searchable_params)
    add_dependent_uniform_param('NUM_CONV_BATCH_NORM_LAYERS', 'NUM_CONV_LAYERS', searchable_params)
    add_dependent_uniform_param('NUM_CONV_DROPOUT_LAYERS', 'NUM_CONV_LAYERS', searchable_params)
    add_dependent_uniform_param('NUM_DENSE_BATCH_NORM_LAYERS', 'NUM_DENSE_LAYERS', searchable_params)
    add_dependent_uniform_param('NUM_DENSE_DROPOUT_LAYERS', 'NUM_DENSE_LAYERS', searchable_params)
    return searchable_params


def update_with_computed_params(cur_params):
    for param_name in ('BATCH_SIZE', 'NUM_CONV_LAYERS', 'NUM_DENSE_LAYERS', 'CONV_KERNEL_SIZE', 'POOL_SIZE', 'INITIAL_CONV_FILTERS', 'INITIAL_DENSE_SIZE', 'MAX_PRETRAIN_EPOCHS', 'MAX_FINETUNE_EPOCHS', 'PRETRAIN_LR_EPOCHS', 'FINETUNE_LR_EPOCHS', 'NUM_POOLING_LAYERS', 'NUM_CONV_BATCH_NORM_LAYERS', 'NUM_CONV_DROPOUT_LAYERS', 'NUM_DENSE_BATCH_NORM_LAYERS', 'NUM_DENSE_DROPOUT_LAYERS'):
        cur_params[param_name] = int(cur_params[param_name])
    cur_params['NAME'] = common.hash_data(cur_params)


def compute_loss(hyperparam_dirname, name):
    results = pd.read_csv(os.path.join('backups', hyperparam_dirname, 'results', f'{name}_0.csv'))
    val_loss_results = results['Val loss']
    val_f1_macro_results = results['Val f1 macro']
    test_f1_macro_results = results['Test f1 macro']
    val_f1_macro_results_before_thresholding = results['Val f1 macro before thresholding']
    test_f1_macro_results_before_thresholding = results['Test f1 macro before thresholding']
    if len(val_loss_results) > 1:
        loss_variance = np.var(val_loss_results, ddof=1)
    else:
        loss_variance = None
    return val_loss_results.mean(), loss_variance, val_f1_macro_results.mean(), test_f1_macro_results.mean(), val_f1_macro_results_before_thresholding.mean(), test_f1_macro_results_before_thresholding.mean()


def run_trial(seed, searchable_params):
    cur_params = CONSTANT_PARAMS.copy()
    hyperparam_dirname = f'{_BASE_HYPERPARAM_DIRNAME}_seed_{seed}'
    cur_params['SEED'] = seed
    cur_params['HYPERPARAM_DIRNAME'] = hyperparam_dirname
    cur_params.update(searchable_params)
    update_with_computed_params(cur_params)
    json_path = os.path.join('backups', hyperparam_dirname, 'params', f'{cur_params["NAME"]}.json')
    log_path = os.path.join('backups', hyperparam_dirname, 'logs', f'{cur_params["NAME"]}.txt')
    with open(json_path, 'wt') as f:
        json.dump(cur_params, f)
    with open(log_path, 'wt') as f:
        result = subprocess.run(['python', 'spectrogram_train_script.py', json_path], stdout=f, stderr=subprocess.STDOUT)
    result_dict = {
        'name': cur_params['NAME']
    }
    if result.returncode == 0:
        result_dict['status'] = hyperopt.STATUS_OK
        loss, loss_variance, val_f1_macro, test_f1_macro, val_f1_macro_before_thresholding, test_f1_macro_before_thresholding = compute_loss(hyperparam_dirname, cur_params['NAME'])
        result_dict['loss'] = loss
        if loss_variance is not None:
            result_dict['loss_variance'] = loss_variance
        result_dict['val_f1_macro'] = val_f1_macro
        result_dict['test_f1_macro'] = test_f1_macro
        result_dict['val_f1_macro_before_thresholding'] = val_f1_macro_before_thresholding
        result_dict['test_f1_macro_before_thresholding'] = test_f1_macro_before_thresholding
    else:
        result_dict['status'] = hyperopt.STATUS_FAIL
        print(f'Run {cur_params["NAME"]} failed. Check the log for more details.')
    print(result_dict)
    return result_dict


# Adapted from https://github.com/hyperopt/hyperopt/issues/267#issuecomment-272718758
def run_trials(seed, space):
    hyperparam_dirname = f'{_BASE_HYPERPARAM_DIRNAME}_seed_{seed}'
    trials_path = os.path.join('backups', hyperparam_dirname, 'hyperopt_trials', 'hyperopt.pkl')
    trials_step = 1  # how many additional trials to do after loading saved trials. 1 = save after iteration
    max_trials = 20  # initial max_trials. put something small to not have to wait
    try:  # try to load an already saved trials object, and increase the max
        with open(trials_path, 'rb') as f:
            trials = pickle.load(f)
        print('Found saved Trials! Loading...')
        max_trials = len(trials.trials) + trials_step
        print('Rerunning from {} trials to {} (+{}) trials'.format(len(trials.trials), max_trials, trials_step))
    except:  # create a new trials object and start searching
        trials = hyperopt.Trials()

    best = hyperopt.fmin(fn=lambda params: run_trial(seed, params), space=space, algo=hyperopt.tpe.suggest, max_evals=max_trials, trials=trials)

    print('Best:', best)
    
    # save the trials object
    with open(trials_path, 'wb') as f:
        pickle.dump(trials, f)


def main():
    space = create_searchable_params_space()
    # First run 20 trials for each seed to finish the random exploration stage.
    # Then continue to run 1 trial for each seed until the program stops.
    # If the program is restarted, it will continue to run 20 trials for each seed.
    while True:
        for seed in (100, 200, 300):
            run_trials(seed, space)


if __name__ == '__main__':
    main()