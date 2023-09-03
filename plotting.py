import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_results(overall_performance, save_path="figs/"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for course_name, performance in overall_performance.items():
        for clf_name, clf_data in performance.items():
            # plot Confusion Matrix
            y_test = clf_data["y_test"]
            y_pred = clf_data["y_pred"]
            cm = confusion_matrix(y_test, y_pred)

            plt.figure(figsize=(6, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix: {clf_name} on {course_name}")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.savefig(f"{save_path}/cm_{course_name}_{clf_name}.png")
            plt.close()

            # plot Learning Curves
            lc_data = clf_data["learning_curve"]
            train_sizes = lc_data["train_sizes"]
            train_scores = lc_data["train_scores"]
            test_scores = lc_data["test_scores"]

            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)

            plt.figure(figsize=(8, 6))
            plt.plot(train_sizes, train_mean, label="Training score", color="blue")
            plt.fill_between(
                train_sizes,
                train_mean - train_std,
                train_mean + train_std,
                color="lightblue",
            )
            plt.plot(
                train_sizes, test_mean, label="Cross-validation score", color="red"
            )
            plt.fill_between(
                train_sizes,
                test_mean - test_std,
                test_mean + test_std,
                color="lightcoral",
            )
            plt.title(f"Learning Curve: {clf_name} on {course_name}")
            plt.xlabel("Training Set Size")
            plt.ylabel("Accuracy Score")
            plt.legend(loc="best")
            plt.savefig(f"{save_path}/lc_{course_name}_{clf_name}.png")
            plt.close()
