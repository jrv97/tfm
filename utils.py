import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def get_labels_counts(target):
    """
    Retrieve unique labels and their respective counts from the provided target data.

    Parameters:
    - target (np.ndarray or pd.Series): An array or series containing target values or labels.

    Returns:
    - tuple: A tuple containing two elements:
        1. unique (np.ndarray or pd.Index): An array or index of unique labels or values.
        2. counts (np.ndarray): An array of counts corresponding to each unique label or value.

    Notes:
    - If `target` is a numpy ndarray, the function uses numpy's unique method to derive counts.
    - If `target` is a pandas Series, the function uses pandas' value_counts method to derive counts.
    """
    if isinstance(target, np.ndarray):
        unique, counts = np.unique(target, return_counts=True)
    else:
        unique = target.value_counts().index
        counts = target.value_counts().values
    return unique, counts


def load_data(path, target_col):
    df = pd.read_csv(path)
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Encoding target if it's of object type
    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)

    return X, y


def split_data(X, y, test_size=0.3, random_state=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def save_to_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_from_pickle(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data
