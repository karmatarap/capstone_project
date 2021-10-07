import math
import os
import random
import sys
from collections import Counter
from typing import NamedTuple, Iterable, Iterator, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torchaudio
from torchaudio import transforms

from PIL import Image
from sklearn import metrics, model_selection, preprocessing
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.sampler import WeightedRandomSampler

from utils.common import load_dz_data
from utils.cross_validation import CrossValidator
from utils.fastai_utils import MLFlowExperiment, MLFlowTracking, set_seeds
from utils.metrics import Metrics
from utils.test_split import TestSplitter


class AudioUtil:
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)

    # ----------------------------
    # Convert the given audio to the desired number of channels
    # ----------------------------
    @staticmethod
    def rechannel(aud, new_channel):
        sig, sr = aud

        if sig.shape[0] == new_channel:
            # Nothing to do
            return aud

        if new_channel == 1:
            # Convert from stereo to mono by selecting only the first channel
            resig = sig[:1, :]
        else:
            # Convert from mono to stereo by duplicating the first channel
            resig = torch.cat([sig, sig])

        return (resig, sr)

    # ----------------------------
    # Since Resample applies to a single channel, we resample one channel at a time
    # ----------------------------
    @staticmethod
    def resample(aud, newsr):
        sig, sr = aud

        if sr == newsr:
            # Nothing to do
            return aud

        num_channels = sig.shape[0]
        # Resample first channel
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1, :])
        if num_channels > 1:
            # Resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:, :])
            resig = torch.cat([resig, retwo])

        return (resig, newsr)
        # --------------------------

    # Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr // 1000 * max_ms

        if sig_len > max_len:
            # Truncate the signal to the given length
            sig = sig[:, :max_len]

        elif sig_len < max_len:
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)

        return (sig, sr)

    # ----------------------------
    # Shifts the signal to the left or right by some percent. Values at the end
    # are 'wrapped around' to the start of the transformed signal.
    # ----------------------------
    @staticmethod
    def time_shift(aud, shift_limit):
        sig, sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)

    # ----------------------------
    # Generate a Spectrogram
    # ----------------------------
    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        sig, sr = aud
        top_db = 80

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = transforms.MelSpectrogram(
            sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels
        )(sig)

        # Convert to decibels
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return spec

    # ----------------------------
    # Augment the Spectrogram by masking out some sections of it in both the frequency
    # dimension (ie. horizontal bars) and the time dimension (vertical bars) to prevent
    # overfitting and to help the model generalise better. The masked sections are
    # replaced with the mean value.
    # ----------------------------
    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(
                aug_spec, mask_value
            )

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return aug_spec


class ElephantCallDS:
    def __init__(self, wav_paths, labels, resize=None, augmentations=None, valid=False):
        self.wav_paths = wav_paths
        self.channel = 2
        self.labels = labels
        self.duration = 15739
        self.sr = 22050  # Max duration of wav file in milliseconds
        self.shift_pct = 0.4
        self.valid = valid

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):

        wav_file = self.wav_paths[idx]
        label = self.labels[idx]

        aud = AudioUtil.open(wav_file)

        if not self.valid:

            reaud = AudioUtil.resample(aud, self.sr)
            rechan = AudioUtil.rechannel(reaud, self.channel)

            dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
            shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
            sgram = AudioUtil.spectro_gram(
                shift_aud, n_mels=102, n_fft=1024, hop_len=None
            )
            aug_sgram = AudioUtil.spectro_augment(
                sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2
            )

        else:
            aug_sgram = AudioUtil.spectro_gram(
                aud, n_mels=128, n_fft=1024, hop_len=None
            )
        return aug_sgram, label


class AudioClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Second Convolution Block
        self.conv3 = nn.Conv2d(
            16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        )
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Second Convolution Block
        self.conv4 = nn.Conv2d(
            32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        )
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=2)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)

    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)

        # Final output
        return x


def train_one_step(model, data, optimizer, criterion, device):
    inputs, labels = data[0].to(device), data[1].to(device)

    # Normalize the inputs
    inputs_m, inputs_s = inputs.mean(), inputs.std()
    inputs = (inputs - inputs_m) / inputs_s

    # Zero the parameter gradients
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss


def train_one_epoch(model, data_loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    for batch_index, data in enumerate(data_loader):
        loss = train_one_step(model, data, optimizer, criterion, device)
        scheduler.step()
        total_loss += loss
    return total_loss


def validate_one_step(model, data, criterion, device):
    inputs, labels = data[0].to(device), data[1].to(device)
    inputs_m, inputs_s = inputs.mean(), inputs.std()
    inputs = (inputs - inputs_m) / inputs_s

    outputs = model(inputs)
    loss = criterion(outputs, labels)
    outputs = torch.argmax(outputs, axis=1)
    return (
        loss,
        labels.detach().cpu().numpy().tolist(),
        outputs.detach().cpu().numpy().tolist(),
    )


def validate_one_epoch(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    final_labels, final_outputs = [], []
    for batch_index, data in enumerate(data_loader):
        with torch.no_grad():
            loss, label, output = validate_one_step(model, data, criterion, device)
        total_loss += loss
        final_labels.extend(label)
        final_outputs.extend(output)
    return total_loss, final_labels, final_outputs


def fold_gen(df, output_path: str, nfolds: int) -> Iterator[Tuple[np.array, np.array]]:
    """Yield training and validation indices per fold"""
    for fold in range(nfolds):
        with open(os.path.join(output_path, f"train_indices_{fold}.csv"), "rt") as f:
            train_indices = np.array([int(index) for index in f.readlines()])
        with open(os.path.join(output_path, f"val_indices_{fold}.csv"), "rt") as f:
            val_indices = np.array([int(index) for index in f.readlines()])

        df_train = df[df.index.isin(train_indices)].reset_index(drop=True)
        df_val = df[df.index.isin(train_indices)].reset_index(drop=True)

        yield df_train, df_val


def training(
    model, train_dl, valid_dl, num_epochs, device, fold=0, learning_rate=0.001
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        steps_per_epoch=int(len(train_dl)),
        epochs=num_epochs,
        anneal_strategy="linear",
    )

    # Repeat for each epoch
    min_valid_loss = math.inf
    for epoch in range(num_epochs):

        train_loss = train_one_epoch(
            model, train_dl, optimizer, scheduler, criterion, device
        )
        valid_loss, labels, outputs = validate_one_epoch(
            model, valid_dl, criterion, device
        )
        f1 = metrics.f1_score(labels, outputs, average="macro")
        print(
            f"Training Loss: {train_loss}, Validation Loss: {valid_loss}, F1Score: {f1}"
        )
        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            print("Saving model.....")
            torch.save(model.state_dict(), f"{fold}-best-model-parameters.pt")


def main():
    k_folds = 5
    num_epochs = 10
    data_params = {
        "STRATIFY_COL": "agecat",
        "NUM_K_FOLDS": k_folds,
        "BASE_DATA_DIR": "./data/metadata",
        "OUTPUT_PATH": "./data/metadata",
        "SPECTROGRAM_DIR": "./data/metadata/spectrograms",
        "SEED": 42,
    }
    df = load_dz_data(data_params["BASE_DATA_DIR"], target_col="agecat")
    df["wav_path"] = "./data/raw/" + df["unique_ID"] + ".wav"
    lbl_enc = preprocessing.LabelEncoder()
    for fold, (df_train, df_valid) in enumerate(
        fold_gen(df, data_params["OUTPUT_PATH"], nfolds=k_folds)
    ):

        # Print
        print(f"FOLD {fold}")
        print("--------------------------------")

        print(set(df_train.agecat))
        train_targets = lbl_enc.fit_transform(df_train.agecat)

        class_sample_count = np.array(list(Counter(train_targets).values()))
        weight = 1.0 / class_sample_count
        samples_weight = np.array([weight[t] for t in train_targets])

        samples_weight = torch.from_numpy(samples_weight)

        train_sampler = WeightedRandomSampler(
            samples_weight.type("torch.DoubleTensor"), len(samples_weight)
        )

        train_dataset = ElephantCallDS(df_train.wav_path, train_targets)
        train_dl = torch.utils.data.DataLoader(
            train_dataset, batch_size=12, sampler=train_sampler
        )

        valid_targets = lbl_enc.fit_transform(df_valid.agecat)
        valid_dataset = ElephantCallDS(df_valid.wav_path, valid_targets)
        valid_dl = torch.utils.data.DataLoader(
            valid_dataset, batch_size=12, shuffle=False
        )

        myModel = AudioClassifier()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        myModel = myModel.to(device)
        training(myModel, train_dl, valid_dl, 10, device, fold=fold)


if __name__ == "__main__":
    main()
