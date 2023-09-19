import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.ensemble import ExtraTreesClassifier, IsolationForest
from sklearn.feature_selection import (RFE, SelectKBest, VarianceThreshold,
                                       mutual_info_classif)
from sklearn.neighbors import LocalOutlierFactor
from xgboost import XGBClassifier


def remove_outliers_isolation_forest(
    X_train, y_train, X_test, y_test, K=10, top_feats=10, plot=True
):
    from sklearn.ensemble import IsolationForest

    isf = IsolationForest(n_jobs=-1, random_state=1)
    isf.fit(X_train)
    predictions = isf.predict(X_train)
    # The predict method returns 1 for inliers and -1 for outliers.
    X_train_filtered = X_train[predictions == 1]
    y_train_filtered = y_train[predictions == 1]
    return X_train_filtered, y_train_filtered, X_test, y_test


def remove_outliers_zscore(X_train, y_train, threshold=3):
    z_scores = np.abs(zscore(X_train))
    valid_samples = (z_scores < threshold).all(axis=1)
    return X_train[valid_samples], y_train[valid_samples]


def remove_outliers_iqr(X_train, y_train, k=1.5):
    Q1 = X_train.quantile(0.25)
    Q3 = X_train.quantile(0.75)
    IQR = Q3 - Q1
    valid_samples = ((X_train >= (Q1 - k * IQR)) & (X_train <= (Q3 + k * IQR))).all(
        axis=1
    )
    return X_train[valid_samples], y_train[valid_samples]


def remove_outliers_dbscan(X_train, y_train, eps=0.5, min_samples=5):
    from sklearn.cluster import DBSCAN

    outlier_detection = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = outlier_detection.fit_predict(X_train)
    valid_samples = clusters != -1
    return X_train[valid_samples], y_train[valid_samples]


def remove_outliers_lof(X_train, y_train, n_neighbors=20):
    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    y_pred = lof.fit_predict(X_train)
    valid_samples = y_pred != -1
    return X_train[valid_samples], y_train[valid_samples]


def remove_outliers_isolation_forest(X_train, y_train, contamination=0.05):
    iso_forest = IsolationForest(contamination=contamination)
    y_pred = iso_forest.fit_predict(X_train)
    valid_samples = y_pred != -1
    return X_train[valid_samples], y_train[valid_samples]


def features_selection_rfe(
    X_train, y_train, X_test, y_test, K=10, top_feats=10, plot=True
):
    rfe = RFE(estimator=XGBClassifier(n_jobs=-1, random_state=1))
    rfe.fit(X_train, y_train)

    selected_features = X_train.columns[rfe.support_].tolist()

    if plot:
        print("Selected Features by RFE:")
        print(selected_features)
        print(len(selected_features))

    return X_train[rfe.support_], y_train, X_test[rfe.support_], y_test


def features_selection_extratrees(
    X_train, y_train, X_test, y_test, K=10, top_feats=10, plot=True
):
    forest = ExtraTreesClassifier(n_estimators=250, max_depth=5, random_state=1)
    forest.fit(X_train, y_train)

    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    indices = indices[:top_feats]

    if plot:
        plt.figure(figsize=(12, 6))
        plt.title("Top Feature Importances")
        plt.bar(
            range(top_feats),
            importances[indices],
            yerr=std[indices],
            align="center",
            alpha=0.7,
        )
        plt.xticks(range(top_feats), X_train.columns[indices], rotation=90)
        plt.xlabel("Features")
        plt.ylabel("Importance")
        plt.show()

    return X_train[indices], y_train, X_test[indices], y_test


def features_selection_variance_threshold(
    X_train, y_train, X_test, y_test, K=10, top_feats=10, plot=True
):
    # Apply Variance Threshold
    var_thr = VarianceThreshold(threshold=0.0)
    var_thr.fit(X_train)

    return (
        X_train[X_train.columns[var_thr.get_support()]],
        y_train,
        X_test[X_test.columns[var_thr.get_support()]],
        y_test,
    )


def features_selection_mutual_information(
    X_train, y_train, X_test, y_test, K=10, top_feats=10, plot=True
):
    # Compute Mutual Information
    mf = mutual_info_classif(X_train, y_train)
    mf = pd.Series(mf, index=X_train.columns)
    if plot:
        mf.sort_values(ascending=False).plot(kind="bar", figsize=(14, 7))
        plt.show()
    # Select the top K features based on Mutual Information
    top_col = SelectKBest(mutual_info_classif, k=K)
    top_col.fit(X_train, y_train)
    selected_features = X_train.columns[top_col.get_support()]

    return X_train[selected_features], y_train, X_test[selected_features], y_test


def smote(X_train, y_train, X_test, y_test, K=10, top_feats=10, plot=True):
    return X_train, y_train, X_test, y_test


def svm_smote(X_train, y_train, X_test, y_test, K=10, top_feats=10, plot=True):
    return X_train, y_train, X_test, y_test


features_selection = [
    features_selection_extratrees,
    features_selection_mutual_information,
    features_selection_rfe,
    features_selection_variance_threshold,
]

oversampling = [smote, svm_smote]

remove_outliers = [remove_outliers_isolation_forest]


ALL_PREPROCESSING_OPTIONS = [features_selection, oversampling, remove_outliers]


def get_all_configurations():
    for r in range(1, len(ALL_PREPROCESSING_OPTIONS) + 1):
        for subset in itertools.combinations(ALL_PREPROCESSING_OPTIONS, r):
            all_combinations = list(itertools.product(*subset))

            for combination in all_combinations:
                yield combination


PREPROCESSING_CONFIGURATIONS = list(get_all_configurations())