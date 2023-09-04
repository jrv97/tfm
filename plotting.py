import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LearningCurveDisplay


def plot_results(overall_performance, save_path="results/figs/"):
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

            display = LearningCurveDisplay(
                train_scores=train_scores,
                test_scores=test_scores,
                train_sizes=train_sizes,
            )
            display.plot()
            plt.title(f"Learning Curve: {clf_name} on {course_name}")
            plt.savefig(f"{save_path}/lc_{course_name}_{clf_name}.png")
            plt.close()
