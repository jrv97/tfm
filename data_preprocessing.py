import os

import pandas as pd

from config import IGNORED_FEAT, TARGET
from oversampling import oversample_data


def load_datasets(input_folder, oversampling=None):
    datasets = {}

    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            course_name = filename[:-4]
            df = pd.read_csv(f"{input_folder}/{filename}")
            clean = clean_dataset(df)
            datasets[course_name] = clean
            if oversampling is not None:
                datasets[course_name] = augment_data(clean, oversampling)

    return datasets


def clean_dataset(df):
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    df = df.drop(IGNORED_FEAT, axis=1)
    df[TARGET] = le.fit_transform(df[TARGET])
    return df


def augment_data(df, oversampling_technique):
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    X_res, y_res = oversample_data(X, y, oversampling_technique)
    df_resampled = pd.concat(
        [pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=TARGET)], axis=1
    )
    return df_resampled
