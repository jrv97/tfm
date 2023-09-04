import pandas as pd
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    mutual_info_classif,
)


def apply_variance_threshold(X_train, threshold=0.0):
    var_thr = VarianceThreshold(threshold=threshold)
    var_thr.fit(X_train)
    X_train_filtered = X_train[X_train.columns[var_thr.get_support()]]
    return X_train_filtered


def compute_mutual_info(X_train, y_train):
    mf = mutual_info_classif(X_train, y_train)
    mutual_info = pd.Series(mf, index=X_train.columns)
    return mutual_info


def select_k_best_features(X_train, y_train, k=10):
    top_col = SelectKBest(mutual_info_classif, k=k)
    top_col.fit(X_train, y_train)
    selected_features = X_train.columns[top_col.get_support()]
    return X_train[selected_features]
