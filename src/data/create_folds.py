import os
import numpy as np
from utils.cross_validation import CrossValidator

if __name__ == "__main__":
    PARAMS = {
        "BASE_DATA_DIR": "./data/metadata",
        "NUM_K_FOLDS": 5,
        "SEED": 100,
        "STRATIFY_COL": "agecat",
        "OUTPUT_PATH": "./data/metadata",
    }
    train_val_indices_filename = os.path.join(
        PARAMS["OUTPUT_PATH"], "train_val_indices.csv"
    )
    with open(train_val_indices_filename, "rt") as f:
        train_val_indices = np.array([int(index) for index in f.readlines()])
    CrossValidator(PARAMS).get_no_leakage_crossval_splits(train_val_indices)
