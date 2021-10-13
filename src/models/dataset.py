import librosa
import numpy as np
import cv2
import torch
from audiomentations import SpecFrequencyMask


class ElephantDataset:
    def __init__(
        self, wav_paths, labels, params, wav_augmentations=None, spec_augmentations=None
    ):
        self.wav_paths = wav_paths
        self.labels = labels

        self.params = params

        self.wav_augmentations = wav_augmentations
        self.spec_augmentations = spec_augmentations

    def process(self, wav_file):

        # read wav
        y, sr = librosa.load(wav_file)

        # apply wav augmentations
        if self.wav_augmentations is not None:
            y = self.wav_augmentations(y, self.params.sr)

        # convert to mel spectrogram
        image = librosa.feature.melspectrogram(
            y,
            sr=self.params.sr,
            n_mels=self.params.n_mels,
            fmin=self.params.fmin,
            fmax=self.params.fmax,
            n_fft=self.params.n_fft,
            hop_length=self.params.hop_length,
        )

        image = librosa.power_to_db(image).astype(np.float32)

        # Add Frequency masking, randomly mask rows and columns
        # of spectrogram and impute to mean value
        if self.spec_augmentations is not None:
            image = self.spec_augmentations(image)

        if self.params.resize is not None:
            image = ElephantDataset.resize(image, self.params.resize)

        # Spectrogram has no colour channel, required for CNNs
        image = ElephantDataset.spec_to_image(image)

        # Pytorch expects CxHxW format
        image = np.transpose(image, (1, 0, 2)).astype(np.float32)

        return image

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        wav_file = self.wav_paths[idx]
        label = self.labels[idx]
        image = self.process(wav_file)

        return (
            torch.tensor(image, dtype=torch.float),
            torch.tensor(label, dtype=torch.long),
        )

    @staticmethod
    def spec_to_image(spec, eps=1e6):
        """ Convert mono to color by duplicating channels and normalizing """
        spec = np.stack([spec, spec, spec], axis=1)
        mean = spec.mean()
        std = spec.std()
        spec_norm = (spec - mean) / (std + eps)
        spec_min, spec_max = spec_norm.min(), spec_norm.max()
        spec_norm = np.clip(spec_norm, spec_min, spec_max)
        spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
        spec_scaled = spec_scaled.astype(np.uint8)
        return spec_scaled

    @staticmethod
    def resize(image, size=None):
        if size is not None:
            image = cv2.resize(image, size)
        return image

