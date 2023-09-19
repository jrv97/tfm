# %%
import pickle
import warnings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from xgboost import XGBClassifier

from config import *
from preprocessing import PREPROCESSING_CONFIGURATIONS

warnings.simplefilter(action="ignore", category=Warning)


def _generate_config_key(config):
    return " + ".join(func.__name__ for func in config)


def _prepare_data(
    datapath,
    target_label,
    columns_to_ignore=None,
    labels_to_ignore=None,
    test_size=0.2,
):
    # Read the dataset
    df = pd.read_csv(datapath)

    # Drop specified columns
    if columns_to_ignore:
        df.drop(columns=columns_to_ignore, inplace=True)

    # Drop rows with invalid categories in the target label
    if labels_to_ignore:
        df = df[~df[target_label].isin(labels_to_ignore)]

    # Check if target variable is categorical and convert to numerical if true
    if df[target_label].dtype == "object":
        le = LabelEncoder()
        df[target_label] = le.fit_transform(df[target_label])

    # Check for columns with all NaN values
    cols_all_nan = df.columns[df.isna().all()].tolist()
    if cols_all_nan:
        df.drop(columns=cols_all_nan, inplace=True)
        print(f"Columns {cols_all_nan} were dropped because they contained only NaNs.")

    # Handle missing values - Option 1: Drop them
    # df.dropna(inplace=True)

    # Handle missing values - Option 2: Impute them (in this case, with mean)
    imputer = SimpleImputer(strategy="mean")
    imputed_data = imputer.fit_transform(df)
    df = pd.DataFrame(imputed_data, columns=df.columns)

    # Split features and target variable
    X = df.drop(columns=target_label)
    y = df[target_label]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    return X_train, y_train, X_test, y_test


def _apply_techniques(config, X_train, y_train, X_test, y_test):
    outliers_detection_technique = config[0]
    features_selection_technique = config[1]
    oversampling_technique = config[2]

    # Apply outlier removal
    X_train, y_train = outliers_detection_technique(X_train, y_train)

    # Apply features selection
    X_train, y_train, X_test, y_test = features_selection_technique(
        X_train, y_train, X_test, y_test
    )

    # Apply oversampling
    X_train, y_train = oversampling_technique(X_train, y_train)

    return X_train, y_train, X_test, y_test


def get_data_for_config(
    config,
    datapath,
    target,
    columns_to_ignore=None,
    labels_to_ignore=None,
    test_size=0.2,
):
    X_train, y_train, X_test, y_test = _prepare_data(
        datapath=datapath,
        columns_to_ignore=columns_to_ignore,
        target_label=target,
        labels_to_ignore=labels_to_ignore,
        test_size=test_size,
    )
    X_train, y_train, X_test, y_test = _apply_techniques(
        config, X_train, y_train, X_test, y_test
    )
    return X_train, y_train, X_test, y_test


# %%
def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


# Training function
def train_classifiers(X_train, y_train, X_test, y_test, config_key):
    results = {}
    for classifier_name, classifier_info in classifiers.items():
        try:
            clf = classifier_info["model"]
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)
            if hasattr(clf, "predict_proba"):
                probabilities = clf.predict_proba(X_test)[
                    :, 1
                ]  # Probabilities for the positive class
            else:
                probabilities = predictions  # For models like SVM without predict_proba, use predictions
            metrics = compute_metrics(y_test, predictions, probabilities)

            results[classifier_name] = {
                "metrics": metrics,
                "data": {
                    "X_train": X_train,
                    "y_train": y_train,
                    "X_test": X_test,
                    "y_test": y_test,
                    "y_pred": predictions,
                },
            }
        except Exception as e:
            print(
                f"{classifier_name} failed to train with configuration {config_key} because: {e}"
            )
    return results


classifiers = {
    "RandomForest": {
        "model": RandomForestClassifier(),
    },
    "K-nearest-neighbor": {
        "model": KNeighborsClassifier(),
    },
    "Artificial Neural Network": {
        "model": MLPClassifier(),
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(),
    },
    "Logistic Regression": {
        "model": LogisticRegression(),
    },
    "Support Vector Machine": {
        "model": SVC(),  # https: //www.kaggle.com/code/sunayanagawde/ml-algorithms-usage-and-prediction?scriptVersionId=120249289&cellId=62
    },
    "Naive Bayes": {
        "model": GaussianNB(),
    },
    "XG-boost": {
        "model": XGBClassifier(),
    },
}

kaggle = (
    "kaggle",
    KAGGLE_DATA_PATH,
    KAGGLE_IGNORED_FEAT,
    KAGGLE_IGNORED_LABELS,
    KAGGLE_TARGET,
    0.8,
)
moodle = (
    "moodle",
    MOODLE_DATA_PATH,
    MOODLE_IGNORED_FEAT,
    MOODLE_IGNORED_LABELS,
    MOODLE_TARGET,
    0.2,
)
student_pred = (
    "student_pred",
    STUDENT_PRED_PATH,
    STUDENT_PRED_IGNORED_FEAT,
    STUDENT_PRED_IGNORED_LABELS,
    STUDENT_PRED_TARGET,
    0.2,
)
datasets = [
    # kaggle,
    # moodle,
    student_pred,
]

# Main dictionary to store results
all_results = {}

for dataset in datasets:
    dataset_name = dataset[0]
    datapath = dataset[1]
    columns_to_ignore = dataset[2]
    labels_to_ignore = dataset[3]
    target_label = dataset[4]
    test_size = dataset[5]
    for config in tqdm(PREPROCESSING_CONFIGURATIONS, desc=f"Dataset: {dataset_name}"):
        config_key = _generate_config_key(config)
        try:
            X_train, y_train, X_test, y_test = get_data_for_config(
                config,
                datapath=datapath,
                target=target_label,
                columns_to_ignore=columns_to_ignore,
                labels_to_ignore=labels_to_ignore,
                test_size=test_size,
            )
            all_results[config_key] = train_classifiers(
                X_train, y_train, X_test, y_test, config_key
            )
        except Exception as e:
            print(f"{config_key} is invalid for dataset {dataset_name} because: {e}")
        # Serialize results
        with open(f"results/all_results_{dataset_name}.pkl", "wb") as f:
            pickle.dump(all_results, f)

# %% [markdown]
# # TO-DO
#
# - implementar cross-validation
# - ✅ limpiar los NaN de los datasets
# - ✅ en feat_selection se usa k=10. cambiar esto a tomar los mas relevantes calculando k en funcion de cada escenario Y/O hacer k = min(n_feat, 10) pq hay datasets que tienen menos de 10 feats
# - añadir modelos: algo de ensemble (boosting y bagging) y red neuronal mas turbia
# - graficas para los mejores resultados de cada dataset
# - ✅ añadir roc-auc a las metricas
# - plotear graficas de aprendizaje (learning curves) o guardar info para generarlas despues para los mejores modelos o algo asi
