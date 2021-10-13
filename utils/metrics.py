from datetime import datetime
from typing import List

import pandas as pd
from IPython.display import display
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class Metrics:
    """ Helper class to define metrics to capture """

    def __init__(
        self, targets: List[int], preds: List[int], labels: List[str] = None
    ) -> None:
        self.targets = targets
        self.preds = preds
        self.labels = labels

    def plot_confusion_matrix(self) -> None:
        cm = confusion_matrix(y_true=self.targets, y_pred=self.preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.labels)
        disp.plot()

    def get_metrics_dict(self, prefix: str = "", precision: int = 3) -> dict:
        """ Returns a dict of useful classification metrics by leveraging
            sklearns classification_report. Format is massaged for
            easier logging.
        """
        report = classification_report(
            y_true=self.targets,
            y_pred=self.preds,
            target_names=self.labels,
            output_dict=True,
        )

        metrics_dict = {}
        for i, v in report.items():
            if isinstance(v, dict):
                for m in v:
                    metrics_dict[f"{prefix}_{i}_{m}"] = round(v[m], precision)
            else:
                metrics_dict[f"{prefix}_{i}"] = round(v, precision)
        return metrics_dict

    def score(self, average_="weighted") -> dict:
        """ Brief metrics of interest """
        return {
            "Accuracy": round(accuracy_score(self.targets, self.preds), 3),
            "Balanced_Accuracy": round(
                balanced_accuracy_score(self.targets, self.preds), 3
            ),
            f"F1_{average_}": round(
                f1_score(self.targets, self.preds, average=average_), 3
            ),
            f"Precision_{average_}": round(
                precision_score(self.targets, self.preds, average=average_), 3
            ),
            f"Recall_{average_}": round(
                recall_score(self.targets, self.preds, average=average_), 3
            ),
        }


class ExperimentLogger:
    """ Helper class to write metrics to results.csv
    
    Results.csv consists of the following columns


    Model 	           | Name of model based on task, eg demog_2_class_resnet124
    Datetime 	       | Last datetime model was run
    Model_id 	       | Unique ID of the model created from model name and timestamp
    Notes 	           | What makes this model interestingly different?
    Accuracy 	       | Unbalanced accuracy, in multiclass settings this will be abnormally low
    Balanced_Accuracy  | Weighted to take into consideration class imbalances
    Weighted_F1 	   | F1 score weighted for class imbalance
    Weighted_Precision | Precision weighted for class imbalance
    Weighted_Recall    | Recall weighted for class imbalance


    >>> results = ExperimentLogger("baseline-models-4-age-class", x,z,notes="Test Save")
    
    To preview results
    >>>results.view_current_results()

    How does it compare to the other models
    >>> results.view_model_history()

    Save the results
    >>> results.log_result()

    Get name to save model as
    >>> print(results.model_id)
    """

    def __init__(
        self,
        model_name,
        targets=None,
        preds=None,
        experiments_path="../reports/experiments.csv",
        notes="",
    ):
        """
        Args:
            model_name: Name of model without spaces, must not have multiple names for the same model
            targets: list or numpy array of true test set values
            preds: list or numpy array of predicted test set values
            experiments_path: relative path the experiments.csv for logging
            notes: Explain what was done differently in this model
        """
        self.model_name = model_name
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.model_id = self.model_name + "_" + self.timestamp
        self.notes = notes
        self.experiments_path = experiments_path
        self.experiments = pd.read_csv(experiments_path)
        results = Metrics(targets, preds).score()
        attrs = {
            "Model": self.model_name,
            "Datetime": self.timestamp,
            "Model_id": self.model_id,
            "Notes": self.notes,
        }
        attrs.update(results)
        self.results_df = pd.DataFrame(attrs, index=[0])

    def view_current_results(self):
        display(self.results_df)

    def view_all_history(self):
        display(experiments)

    def view_model_history(self):
        display(self.experiments.loc[self.experiments.Model == self.model_name])

    def log_result(self):
        with open(self.experiments_path, mode="a") as f:
            self.results_df.to_csv(f, header=f.tell() == 0, index=False)

