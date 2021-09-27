import os
import sys
from pprint import pprint
from typing import Iterator, Optional, Tuple

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
from fastai.vision.all import (
    CategoryBlock,
    ClassificationInterpretation,
    ColReader,
    ColSplitter,
    DataBlock,
    EarlyStoppingCallback,
    ImageBlock,
    Resize,
    SaveModelCallback,
    accuracy,
    cnn_learner,
    error_rate,
    resnet50,
    resnet101,
    resnet152,
)
from sklearn.metrics import classification_report, confusion_matrix

from utils.fastai_utils import MLFlowExperiment, MLFlowTracking, set_seeds
from utils.metrics import Metrics, get_metrics_dict, plot_confusion_matrix


def get_train_val_indices(PARAMS: dict) -> np.array:
    """Helper function to read training indices"""
    train_val_indices_filename = os.path.join(
        PARAMS["OUTPUT_PATH"], "train_val_indices.csv"
    )
    with open(train_val_indices_filename, "rt") as f:
        train_val_indices = np.array([int(index) for index in f.readlines()])
    return train_val_indices


def get_test_indices(PARAMS: dict) -> np.array:
    """Helper function to read test indices"""
    test_indices_filename = os.path.join(PARAMS["OUTPUT_PATH"], "test_indices.csv")
    with open(test_indices_filename, "rt") as f:
        test_indices = np.array([int(index) for index in f.readlines()])
    return test_indices


def fold_gen(PARAMS: dict) -> Iterator[Tuple[np.array, np.array]]:
    """Yield training and validation indices per fold"""
    nfolds = PARAMS["NUM_K_FOLDS"]
    base_dir = PARAMS["BASE_DATA_DIR"]
    output_path = PARAMS["OUTPUT_PATH"]
    for fold in range(nfolds):
        with open(os.path.join(output_path, f"train_indices_{fold}.csv"), "rt") as f:
            train_indices = np.array([int(index) for index in f.readlines()])
        with open(os.path.join(output_path, f"val_indices_{fold}.csv"), "rt") as f:
            val_indices = np.array([int(index) for index in f.readlines()])
        yield train_indices, val_indices


def train(
    DATA_PARAMS: dict,
    MODEL_PARAMS: dict,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    experiment: MLFlowExperiment,
    validation_results: Optional[bool] = False,
    test_results: Optional[bool] = True,
    register: Optional[bool] = True,
) -> None:
    """ Main Training loop for fastai pre-trained models
printidation results. Defaults to False.
        test_results (optional): Show within fold and aggregated test results. Defaults to True.
        register (optional): log experiment to MLFlow. Defaults to True.
    """

    set_seeds(MODEL_PARAMS["SEED"])

    # Start mlflow experiment
    experiment.register_params(MODEL_PARAMS)

    val_preds, val_targs = [], []
    test_preds, test_targs = [], []

    # Force order of labels so folds are comparable
    labels = list(sorted(set(df_train[MODEL_PARAMS["TARGET"]])))
    for fold, (tr_idx, vl_idx) in enumerate(fold_gen(DATA_PARAMS)):
        print(f"FOLD={fold}================")

        # Create column to specify if validation set
        df_train["is_valid"] = df_train.index.isin(vl_idx)

        # Define a fastai datablock to ingest data
        spectrogramBlock = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            splitter=ColSplitter(),  # Uses "is_valid" column to determine validation set
            get_x=ColReader(0),  # path column
            get_y=ColReader(1),  # label column
            item_tfms=Resize(224),  # resize to expected shape of pretrained model
        )

        dls = spectrogramBlock.dataloaders(
            df_train, bs=MODEL_PARAMS["BATCH_SIZE"], classes=labels
        )

        learn = cnn_learner(
            dls,
            # load pre-trained model
            eval(MODEL_PARAMS["PRETRAINED"]),
            # Monitor accuracy and error rate, error rate is what we really care about
            metrics=[accuracy, error_rate],
            # Callbacks used to register within model metrics to MLFlow
            cbs=[
                MLFlowTracking(
                    metric_names=["valid_loss", "train_loss", "error_rate", "accuracy"],
                    client=experiment.mlfclient,
                    run_id=experiment.mlrun.info.run_uuid,
                )
            ],
        ).to_fp16()  # Training with half precision

        # Train model for n epochs, saving the "best" model
        # "best" is defined as the model with lowest validation loss.
        # Model training is stopped when model does not improve for
        # [patience] epochs.
        learn.fine_tune(
            MODEL_PARAMS["EPOCHS"],
            cbs=[
                EarlyStoppingCallback(
                    monitor="valid_loss", patience=MODEL_PARAMS["PATIENCE"]
                ),
                SaveModelCallback(every_epoch=True, monitor="valid_loss"),
            ],
        )

        # Save best model per fold
        model_name = f"{experiment.experiment_name}-{fold}.pkl"
        learn.export(model_name)
        if register:
            experiment.register_model(model_name)

        # Within fold metrics
        if validation_results:
            interp = ClassificationInterpretation.from_learner(learn)
            val_preds.extend(torch.argmax(interp.preds, dim=1))
            val_targs.extend(interp.targs)

        # Using best fold model on holdout test set and store results
        if test_results:
            test_dls = dls.test_dl(df_test, with_labels=True)
            interp = ClassificationInterpretation.from_learner(learn, dl=test_dls)
            test_preds.extend(torch.argmax(interp.preds, dim=1))
            test_targs.extend(interp.targs)

    print("Summary Results")
    # Calculating aggregated metrics accross folds

    if validation_results:
        val_metrics = get_metrics_dict(
            val_targs, val_preds, labels=labels, prefix="valid"
        )
        pprint(val_metrics)
        plot_confusion_matrix(val_targs, val_preds, labels=labels)
        plt.savefig("validation_confusion.png")

        # log metrics with Mlflow
        if register:
            experiment.register_metrics(val_metrics)
            experiment.register_artifact("validation_confusion.png")

    if test_results:
        test_metrics = get_metrics_dict(
            test_targs, test_preds, labels=labels, prefix="test"
        )
        pprint(test_metrics)
        plot_confusion_matrix(val_targs, val_preds, labels=labels)
        plt.savefig("test_confusion.png")
        # log metrics with Mlflow
        if register:
            experiment.register_metrics(test_metrics)
            experiment.register_artifact("test_confusion.png")
