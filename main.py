import warnings

from config import DATA_PATH, OVERSAMPLING_TECHNIQUE
from data_preprocessing import load_datasets
from eval import evaluate_models

# warnings.simplefilter(action="ignore", category=FutureWarning)
# warnings.simplefilter(action="ignore", category=DeprecationWarning)
warnings.simplefilter(action="ignore")


def main():
    data_path = DATA_PATH
    oversampling_technique = OVERSAMPLING_TECHNIQUE
    datasets = load_datasets(data_path, oversampling_technique)
    evaluate_models(datasets)


if __name__ == "__main__":
    main()
