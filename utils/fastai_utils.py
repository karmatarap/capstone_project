import random
from typing import List

import numpy as np
import torch
from fastai.vision.all import Callback
from mlflow.entities.run import Run
from mlflow.tracking import MlflowClient

"""
Helper classes and functions when using fastai libraries
"""


class MLFlowTracking(Callback):
    "A `LearnerCallback` that tracks the loss and other metrics into MLFlow"

    def __init__(
        self, metric_names: List[str], client: MlflowClient, run_id: Run
    ) -> None:
        self.client = client
        self.run_id = run_id
        self.metric_names = metric_names

    def after_epoch(self) -> None:
        "Compare the last value to the best up to now"
        for metric_name in self.metric_names:
            m_idx = list(self.recorder.metric_names[1:]).index(metric_name)
            if len(self.recorder.values) > 0:
                val = self.recorder.values[-1][m_idx]
                self.client.log_metric(self.run_id, metric_name, np.float(val))


class MLFlowExperiment:
    def __init__(self, experiment_name: str) -> None:
        self.experiment_name = experiment_name

        # Instantiate MLFlow client
        self.mlfclient = MlflowClient()

        # Check if the experiment already exists, or create it
        mlfexp = self.mlfclient.get_experiment_by_name(experiment_name)
        if mlfexp is None:
            mlfexp_id = self.mlfclient.create_experiment(experiment_name)
            mlfexp = self.mlfclient.get_experiment_by_name(experiment_name)

        self.mlrun = self.mlfclient.create_run(experiment_id=mlfexp.experiment_id)

    def register_params(self, params: dict) -> None:
        "Logs params with MLflow"
        for k, v in params.items():
            self.mlfclient.log_param(run_id=self.mlrun.info.run_uuid, key=k, value=v)

    def register_metrics(self, metrics: dict) -> None:
        "Logs metrics with MLflow"
        for k, v in metrics.items():
            self.mlfclient.log_metric(run_id=self.mlrun.info.run_uuid, key=k, value=v)

    def register_artifact(self, artifact_path: str) -> None:
        "Logs local files to MLflow"
        self.mlfclient.log_artifact(
            run_id=self.mlrun.info.run_uuid, local_path=artifact_path
        )

    def register_model(self, model_path: str) -> None:
        # logging fastai models as artifacts for the moment
        self.mlfclient.log_artifact(
            run_id=self.mlrun.info.run_uuid, local_path=model_path
        )


def set_seeds(seed: int) -> None:
    """ Setting seeds in fastai requires setting seed for all possible random number generators"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
