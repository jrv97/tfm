import os

import pandas as pd

from oversampling import oversample_data


def load_datasets(input_folder, ignore_cols, target, oversampling=None):
    datasets = {}

    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            course_name = filename[:-4]
            df = pd.read_csv(f"{input_folder}/{filename}")
            clean = _clean_dataset(df, ignore_cols, target)
            datasets[course_name] = clean
            if oversampling is not None:
                datasets[course_name] = _augment_data(clean, oversampling, target)

    return datasets


def _clean_dataset(df, ignore_cols, target):
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    df = df.drop(ignore_cols, axis=1)
    df[target] = le.fit_transform(df[target])
    return df


def _augment_data(df, oversampling_technique, target):
    X = df.drop(target, axis=1)
    y = df[target]

    X_res, y_res = oversample_data(X, y, oversampling_technique)
    df_resampled = pd.concat(
        [pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=target)], axis=1
    )
    return df_resampled
