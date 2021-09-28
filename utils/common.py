"""Common functions used in multiple different utilities.

Author: Lucy Tan

Exports the load_dz_data function for loading the Dzanga Bai data into a
dataframe.
"""

import os
import numpy as np
import pandas as pd


def load_dz_data(base_data_dir, target_col="age"):
    """Load the Dzanga Bai data from the Excel spreadsheet into a dataframe.

    Also add the spectrogram path, rumble id, and age category columns.

    Additionally, remove any rumbles that are missing age.
    """
    df = pd.read_excel(
        os.path.join(base_data_dir, "Age-sex calls- Dzanga Bai.xlsx"),
        sheet_name="context",
    )
    # Create spectrogram paths in the dataframe.
    df["path"] = df["unique_ID"].apply(
        lambda x: os.path.join(base_data_dir, "spectrograms", f"{x}.png")
    )
    df["exists"] = df["path"].apply(lambda x: os.path.exists(x))
    df["rumble_id"] = df["unique_ID"].apply(lambda x: int(x.split("_")[1]))

    df["agecat"] = df["age"].apply(
        lambda x: "ad/sa"
        if x in ("ad", "sa")
        else "inf/juv"
        if x in ("inf", "juv")
        else "un"
    )
    df = df[df[target_col] != "un"]
    return df


def split_and_stratify_without_leakage(df, seed, split_sizes, stratify_col):
    """Split the data, stratifying by the given column.

    Ensure no data leaks by always adding rumbles with the same id to the same
    split.

    Return the indices for the splits as a tuple of numpy arrays.
    """
    n_splits = len(split_sizes)
    split_indices = [[] for _ in range(n_splits)]

    total_counts_by_target_col = df[stratify_col].value_counts()

    counts_by_target_col = [
        {target: 0 for target in df[stratify_col].unique()} for _ in range(n_splits)
    ]

    rumble_ids_with_indices = df["rumble_id"].reset_index()

    unique_rumble_ids = df["rumble_id"].unique()
    np.random.seed(seed)
    np.random.shuffle(unique_rumble_ids)

    for rumble_id in unique_rumble_ids:
        rumble_id_mask = rumble_ids_with_indices["rumble_id"] == rumble_id
        matching_rumble_indices = rumble_ids_with_indices[rumble_id_mask].index
        # Use the mode of the target column since some groups of rumble
        # ids have different age/sex values.
        # E.g. rumble id 368 has a rumble for "ad" and a rumble for "juv".
        target = df.iloc[matching_rumble_indices][stratify_col].mode()[0]

        # Find which set has the lowest ratio of that target (based on how
        # many it should have).
        ratios = [
            counts_by_target_col[i][target] / split_sizes[i] for i in range(n_splits)
        ]
        chosen_split = np.argmin(ratios)
        split_indices[chosen_split].extend(matching_rumble_indices)
        counts_by_target_col[chosen_split][target] += len(matching_rumble_indices)

    split_indices = tuple(map(np.array, split_indices))
    return split_indices


def output_csv(path, indices):
    """Output a csv with each index as its own line."""
    with open(path, "wt") as f:
        for index in indices:
            f.write(f"{index}\n")

