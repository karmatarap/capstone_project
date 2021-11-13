from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

def wav_to_mel_spectrogram(wav_path: Path, spec_folder: Path=Path("data/spectrograms")) -> None:
    # Passing through arguments to the Mel filters
    y, sr = librosa.load(wav_path)
    S = librosa.feature.melspectrogram(y=y, sr=sr)

    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, sr=sr, ax=ax)
    plt.savefig(spec_folder/f"{wav_path.stem}.png")
    plt.close(fig)
 

if __name__ == "__main__":
    wav_paths = sorted(Path("data/wavs").glob('*.wav'))
    list(map(wav_to_mel_spectrogram,wav_paths))

