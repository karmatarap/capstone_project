import os
import sys
from dataclasses import asdict, dataclass
from pprint import pprint
from typing import Iterator, Optional, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import torch
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

from utils.common import load_dz_data
from utils.cross_validation import CrossValidator
from utils.fastai_utils import MLFlowExperiment, MLFlowTracking, set_seeds
from utils.metrics import Metrics
from utils.test_split import TestSplitter


def get_train_val_indices(output_path: str) -> np.array:
    """Helper function to read training indices"""
    train_val_indices_filename = os.path.join(output_path, "train_val_indices.csv")
    with open(train_val_indices_filename, "rt") as f:
        train_val_indices = np.array([int(index) for index in f.readlines()])
    return train_val_indices


def get_test_indices(output_path: str) -> np.array:
    """Helper function to read test indices"""
    test_indices_filename = os.path.join(output_path, "test_indices.csv")
    with open(test_indices_filename, "rt") as f:
        test_indices = np.array([int(index) for index in f.readlines()])
    return test_indices


def fold_gen(output_path: str, nfolds: int) -> Iterator[Tuple[np.array, np.array]]:
    """Yield training and validation indices per fold"""
    for fold in range(nfolds):
        with open(os.path.join(output_path, f"train_indices_{fold}.csv"), "rt") as f:
            train_indices = np.array([int(index) for index in f.readlines()])
        with open(os.path.join(output_path, f"val_indices_{fold}.csv"), "rt") as f:
            val_indices = np.array([int(index) for index in f.readlines()])
        yield train_indices, val_indices


def create_datasets(data_params: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Handle the creation of train and test datasets

    Returns:
        Train Dataframe, Test Dataframe
    """
    target_col = data_params["STRATIFY_COL"]
    TestSplitter(data_params).get_no_leakage_trainval_test_splits()

    # Create folds
    train_val_indices = get_train_val_indices(data_params["OUTPUT_PATH"])
    CrossValidator(data_params).get_no_leakage_crossval_splits(train_val_indices)
    df = load_dz_data(data_params["BASE_DATA_DIR"], target_col=target_col)

    test_df = df[df.index.isin(get_test_indices(data_params["OUTPUT_PATH"]))]

    # Fastai Dataframes expect train and validation in the same dataframe
    train_val_df = df[df.index.isin(train_val_indices)].reset_index(drop=True)
    df_train = train_val_df[["path", target_col]]
    df_test = test_df[["path", target_col]]
    return df_train, df_test


@dataclass
class ModelParams:
    """ Default and allowed parameters for Models """

    target: str = "age"
    library: str = "fastai"
    epochs: int = 10
    batch_size: int = 10
    transforms: Tuple[str] = ("Resize",)
    batch_transforms: Tuple[str] = None
    patience: int = 3
    pretrained: str = "resnet50"
    num_folds: int = 5
    seed: int = 100


class FastaiModel:
    def __init__(
        self,
        model_params: ModelParams = ModelParams(),
        metadata_path: str = ".",
        model_path: str = ".",
    ) -> None:
        self.model_params = model_params
        self.metadata_path = metadata_path
        self.model_path = model_path

    def train(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        experiment_name: str = "4-class-age",
    ) -> None:
        """ Main training loop 

        Trains models according to options specified in model_params.
        MLFlow logging performed under the provided experiment name
        """
        # Start mlflow experiment
        experiment = MLFlowExperiment(experiment_name)
        # log parameters to mlflow
        experiment.register_params(asdict(self.model_params))

        set_seeds(self.model_params.seed)

        val_preds, val_targs = [], []
        test_preds, test_targs = [], []

        # Force order of labels so folds are comparable
        labels = list(sorted(set(df_train[self.model_params.target])))

        for fold, (tr_idx, vl_idx) in enumerate(
            fold_gen(self.metadata_path, self.model_params.num_folds)
        ):

            # logging.INFO(f"FOLD={fold}================")

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
                df_train, bs=self.model_params.batch_size, classes=labels
            )

            learn = cnn_learner(
                dls,
                # load pre-trained model, to be replaced with non-eval way
                eval(self.model_params.pretrained),
                # Monitor accuracy and error rate, error rate is what we really care about
                metrics=[accuracy, error_rate],
                # Callbacks used to register within model metrics to MLFlow
                cbs=[
                    MLFlowTracking(
                        metric_names=[
                            "valid_loss",
                            "train_loss",
                            "error_rate",
                            "accuracy",
                        ],
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
                self.model_params.epochs,
                cbs=[
                    EarlyStoppingCallback(
                        monitor="valid_loss", patience=self.model_params.patience
                    ),
                    SaveModelCallback(every_epoch=True, monitor="valid_loss"),
                ],
            )

            # Save best model per fold
            model_name = f"{experiment.experiment_name}-{fold}.pkl"
            learn.export(model_name)
            experiment.register_model(model_name)

            # Within fold metrics for validation set
            val_interp = ClassificationInterpretation.from_learner(learn)
            # Convert torch category tensor to numpy and find softmax
            val_output = np.argmax(val_interp.preds.detach().cpu().numpy(), axis=1)

            val_preds.extend(val_output)
            val_targs.extend(val_interp.targs.detach().cpu().numpy())

            # Using best fold model on holdout test set and store results
            test_dls = dls.test_dl(df_test, with_labels=True)
            test_interp = ClassificationInterpretation.from_learner(learn, dl=test_dls)
            test_output = np.argmax(test_interp.preds.detach().cpu().numpy(), axis=1)
            test_preds.extend(test_output)
            test_targs.extend(test_interp.targs.detach().cpu().numpy())

        # Calculating aggregated metrics accross folds
        val_metrics, val_conf_png = FastaiModel.aggregated_metrics(
            val_targs, val_preds, labels=labels, prefix="valid"
        )

        # log metrics with Mlflow
        experiment.register_metrics(val_metrics)
        experiment.register_artifact(val_conf_png)

        # Repeat for test set
        test_metrics, test_conf_png = FastaiModel.aggregated_metrics(
            test_targs, test_preds, labels=labels, prefix="test"
        )
        experiment.register_metrics(test_metrics)
        experiment.register_artifact(test_conf_png)

    @staticmethod
    def aggregated_metrics(
        targs: List[int], preds: List[int], labels: List[str] = None, prefix: str = ""
    ) -> None:
        """ Boilerplate metrics logging """
        m = Metrics(targs, preds, labels=labels)
        m_dict = m.get_metrics_dict(prefix=prefix)
        pprint(m_dict)
        m.plot_confusion_matrix()
        artifact_name = f"{prefix}_confusion.png"
        plt.savefig(artifact_name)
        return m_dict, artifact_name

    def predict(self) -> None:
        pass


def _sample_experiment(target_col="age", experiment_name="4-class-age"):
    """ Example pipeline for experiments """
    for seed in [100, 200, 300]:
        for pretrained in ["resnet50", "resnet101", "resnet152"]:
            for folds in [2, 5, 10]:
                # data params to override
                data_params = {
                    "STRATIFY_COL": target_col,
                    "NUM_K_FOLDS": folds,
                    "BASE_DATA_DIR": "./data/metadata",
                    "OUTPUT_PATH": "./data/metadata",
                    "SPECTROGRAM_DIR": "./data/metadata/spectrograms",
                    "SEED": seed,
                }

                model_params = ModelParams()

                # model parameters to inherit from data_params
                model_params.target = data_params["STRATIFY_COL"]
                model_params.num_folds = data_params["NUM_K_FOLDS"]
                model_params.seed = data_params["SEED"]

                # Model specific parameters

                model_params.pretrained = pretrained
                train_df, test_df = create_datasets(data_params)

                model = FastaiModel(
                    model_params=model_params, metadata_path=data_params["OUTPUT_PATH"]
                )
                model.train(train_df, test_df, experiment_name=experiment_name)


if __name__ == "__main__":
    # _sample_experiment(target_col="age", experiment_name="4-class-age")
    # _sample_experiment(target_col="agecat", experiment_name="2-class-age")
    _sample_experiment(target_col="sex", experiment_name="2-class-sex")

