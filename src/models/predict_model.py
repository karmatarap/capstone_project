import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from utils.common import load_dz_data

from .config import (
    AudioParams,
    best_params,
    data_params,
    spec_aug_combos,
    wav_aug_combos,
)
from .dataset import ElephantDataset
from .engine import Engine
from .models import get_pretrained_model
from .utils import get_test_indices


def predict(model, test_dl):

    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for data in test_dl:
            inputs, labels = data[0].to(device), data[1]
            preds = model(inputs)
            preds = torch.argmax(outputs, axis=1)

        outputs = model(inputs)
        outputs = torch.argmax(outputs, axis=1)
        all_preds.extend(outputs.detach().cpu().numpy().tolist())
        all_labels.extend(labels)
    return all_preds, all_labels


def predict_test():
    target_col = best_params["target"]
    df = load_dz_data(data_params["BASE_DATA_DIR"], target_col=target_col)

    # Create wav paths
    df["wav_path"] = "./data/raw/" + df["unique_ID"] + ".wav"
    test_df = df[df.index.isin(get_test_indices(data_params["OUTPUT_PATH"]))]

    lbl_enc = LabelEncoder()
    test_targets = lbl_enc.fit_transform(test_df[target_col])

    wav_augs = wav_aug_combos[best_params["wav_augs"]]
    spec_augs = spec_aug_combos[best_params["spec_augs"]]
    audio_params = AudioParams()

    test_dataset = ElephantDataset(
        test_df.wav_path,
        test_targets,
        audio_params,
        wav_augmentations=wav_augs,
        spec_augmentations=spec_augs,
    )
    test_dl = torch.utils.data.DataLoader(
        test_dataset, batch_size=12, num_workers=4, shuffle=False
    )

    myModel = get_pretrained_model(best_params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_labels, all_predictions = [], []

    for fold in range(data_params["NUM_K_FOLDS"]):
        myModel.load_state_dict(torch.load(f"{fold}-best-model-parameters.pt"))
        myModel = myModel.to(device)
        p, l = predict(myModel, test_dl)
        all_labels.extend(l)
        all_predictions.extend(p)
    f1 = f1_score(all_labels, all_predictions)
    print(f1)


if __name__ == "__main__":
    predict_test()
