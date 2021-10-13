class AudioParams:
    """ Parameters specific to audio manipulations """

    # Wav params
    channel = 2
    duration = 15739
    sr = 22050

    # Spec params
    n_mels = 512
    fmin = 0
    fmax = sr // 2
    n_fft = sr // 10
    hop_length = sr // (10 * 4)

    # Image params
    resize = 256, 256  # 512,512


data_params = {
    "STRATIFY_COL": "agecat",
    "NUM_K_FOLDS": 5,
    "BASE_DATA_DIR": "./data/metadata",
    "OUTPUT_PATH": "./data/metadata",
    "SPECTROGRAM_DIR": "./data/metadata/spectrograms",
    "SEED": 42,
}

# Placeholder to import best hyperparameters
hyper_params = {}

