from enum import Enum

from imblearn.over_sampling import SVMSMOTE  # SMOTETomek,
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler

from utils import get_labels_counts

# from imblearn.combine import SMOTEENN


class OversamplingTechniques(Enum):
    SMOTE = "smote"
    BORDERLINE_SMOTE = "borderline_smote"
    RANDOM_OVER_SAMPLER = "random_over_sampler"
    # SMOTE_ENN = "smote_enn"
    SVM_SMOTE = "svm_smote"
    # SMOTE_TOMEK = "smote_tomek"


def oversample_data(X, y, technique, n_neighbors_value=None):
    if n_neighbors_value is None:
        _, counts = get_labels_counts(y)
        n_neighbors_value = min(counts) - 1  # One less than the minority class size

    # Choose oversampling technique
    if technique == OversamplingTechniques.SMOTE:
        sampler = SMOTE(random_state=42, k_neighbors=n_neighbors_value)
    elif technique == OversamplingTechniques.BORDERLINE_SMOTE:
        sampler = BorderlineSMOTE(random_state=42)
    elif technique == OversamplingTechniques.RANDOM_OVER_SAMPLER:
        sampler = RandomOverSampler(random_state=42)
    # elif technique == OversamplingTechniques.SMOTE_ENN:
    #     sampler = SMOTEENN(random_state=42)
    elif technique == OversamplingTechniques.SVM_SMOTE:
        sampler = SVMSMOTE(random_state=42, k_neighbors=n_neighbors_value)
    # elif technique == OversamplingTechniques.SMOTE_TOMEK:
    #     sampler = SMOTETomek(random_state=42)
    else:
        raise ValueError("Invalid oversampling technique!")

    X_res, y_res = sampler.fit_resample(X, y)

    return X_res, y_res
