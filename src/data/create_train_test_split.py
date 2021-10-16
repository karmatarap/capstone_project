from utils.test_split import TestSplitter

if __name__ == "__main__":
    PARAMS = {
        "BASE_DATA_DIR": "./data/metadata",
        "NUM_K_FOLDS": 5,
        "SEED": 100,
        "STRATIFY_COL": "agecat",
        "OUTPUT_PATH": "./data/metadata",
    }
    TestSplitter(PARAMS).get_no_leakage_trainval_test_splits()
