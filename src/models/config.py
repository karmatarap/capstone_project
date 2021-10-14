from audiomentations import (
    AddGaussianNoise,
    AddGaussianSNR,
    Compose,
    Normalize,
    PitchShift,
    SpecFrequencyMask,
)


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

best_params = {
    "pretrained_model": "efficientnet-b4",
    "wav_augs": "none",
    "spec_augs": "none",
    "num_layers": 2,
    "hidden_size": 1017,
    "dropout": 0.57,
    "learning_rate": 0.000165,
    "seed": 42,
    "target": "agecat",
    "epochs": 50,
    "batch_size": 12,
}


# Augmentations to be performed directly on the wav files
# ----------------------------------------------------------------
# Normalize: Add a constant amount of gain, normalizes the loudness
# - Should help normalize if elephants are closer or further from detectors
# PitchShift: Changes pitch without changing tempo
# - Would an elephant running cause the calls to change in pitch?
# AddGuassianNoise: Add gaussian noise
# AddGuassianNoiseSNR: Add gaussian noise with random Signal to Noise Ratio
# SpecFrequencyMask: Mask a set of frequencies: see Google AI SpecAugment

# try these combos
# passing mapping as dicts to allow for logging
wav_aug_combos = {
    "none": None,
    "Norm": Normalize(),
    "Norm-SNR": Compose(
        [Normalize(), AddGaussianSNR(min_snr_in_db=0.0, max_snr_in_db=60.0)]
    ),
    "Norm-Gauss-SNR": Compose(
        [
            Normalize(),
            AddGaussianNoise(),
            AddGaussianSNR(min_snr_in_db=0.0, max_snr_in_db=60.0),
        ]
    ),
    "Norm-Gauss-SNR-Pitch": Compose(
        [
            Normalize(),
            AddGaussianNoise(),
            AddGaussianSNR(min_snr_in_db=0.0, max_snr_in_db=60.0),
            PitchShift(),
        ]
    ),
}

spec_aug_combos = {"none": None, "SpecAug": SpecFrequencyMask()}
