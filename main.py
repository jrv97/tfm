import warnings

from config import DATA_PATH, OVERSAMPLING_TECHNIQUE
from data_preprocessing import load_datasets
from eval import evaluate_models
from plotting import (
    plot_confusion_matrices,
    plot_correlation_matrix,
    plot_model_comparison,
)

warnings.simplefilter(action="ignore")


def main():
    data_path = DATA_PATH
    oversampling_technique = OVERSAMPLING_TECHNIQUE
    datasets = load_datasets(data_path, oversampling_technique)
    overall_performance = evaluate_models(datasets)

    # for metric in ["accuracy", "f1_score", "roc_auc"]:
    #     plot_model_comparison(overall_performance, metric=metric)

    # plot_confusion_matrices(overall_performance, datasets)

    # first_course_name = list(datasets.keys())[0]
    # plot_correlation_matrix(datasets[first_course_name])


if __name__ == "__main__":
    main()
