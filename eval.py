import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, learning_curve, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from config import TARGET


def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    auc = roc_auc_score(y_test, y_pred)

    metrics = {"accuracy": accuracy, "f1_score": f1, "roc_auc": auc}

    return metrics


def split_datasets(datasets):
    split_datasets = {}

    for course_name, df in datasets.items():
        X = df.drop(TARGET, axis=1)
        y = df[TARGET]

        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, test_size=0.3, stratify=y, random_state=42
        # )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        split_datasets[course_name] = {
            "x_train": X_train,
            "y_train": y_train,
            "x_test": X_test,
            "y_test": y_test,
        }
    return split_datasets


def evaluate_models(datasets):
    classifiers = {
        "RandomForest": {
            "model": RandomForestClassifier(),
            "params": {  # https://www.kaggle.com/code/sunayanagawde/ml-algorithms-usage-and-prediction?scriptVersionId=120249289&cellId=75
                "bootstrap": [False, True],
                "max_depth": [5, 8, 10, 20],
                "max_features": [3, 4, 5, None],
                "min_samples_split": [2, 10, 12],
                "n_estimators": [100, 200, 300],
                "min_samples_leaf": [1, 2],
                "max_depth": [5, 8, 10, 15, 20],
            },
        },
        "K-nearest-neighbor": {
            "model": KNeighborsClassifier(),
            "params": {
                "n_neighbors": [2, 5],
                "weights": ["uniform", "distance"],
                "leaf_size": [30, 50],
            },
        },
        "Artificial Neural Network": {
            "model": MLPClassifier(),
            "params": {
                "hidden_layer_sizes": [(100,), (50, 50)],
                "activation": ["tanh", "relu"],
                "max_iter": [200, 300, 500, 1000],
            },
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(),
            "params": {
                "criterion": ["gini", "entropy"],
                "splitter": ["best", "random"],
            },
        },
        "Logistic Regression": {
            "model": LogisticRegression(),
            "params": {
                "C": [0.5, 1, 1.5],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear"],
            },
        },
        "Support Vector Machine": {
            "model": SVC(),  # https://www.kaggle.com/code/sunayanagawde/ml-algorithms-usage-and-prediction?scriptVersionId=120249289&cellId=62
            "params": {
                "kernel": ["linear", "rbf"],
                "C": [0.1, 1, 10],
                "gamma": ["scale", "auto"],
            },
        },
        "Naive Bayes": {"model": GaussianNB(), "params": {}},
        "XG-boost": {
            "model": XGBClassifier(),
            "params": {
                "n_estimators": [50, 100],
                "objective": ["binary:logistic"],
                "learning_rate": [0.01, 0.1, 1.0],
            },
        },
    }
    overall_performance = {}
    split = split_datasets(datasets)
    for course_name, data in split.items():
        print(f"Processing data for {course_name}...")

        X_train = data["x_train"]
        y_train = data["y_train"]
        X_test = data["x_test"]
        y_test = data["y_test"]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        course_performance = {}

        for clf_name, clf_spec in classifiers.items():
            print(f"  Training {clf_name}...")

            grid_search = GridSearchCV(
                estimator=clf_spec["model"],
                param_grid=clf_spec["params"],
                cv=5,
                scoring="accuracy",
            )
            grid_search.fit(X_train_scaled, y_train)

            best_params = grid_search.best_params_
            best_model = grid_search.best_estimator_

            y_pred = best_model.predict(X_test_scaled)

            metrics = evaluate_model(y_test, y_pred)
            print(f"  Metrics for {clf_name}: {metrics}")

            # generate learning curves
            train_sizes, train_scores, test_scores = learning_curve(
                best_model,
                X_train_scaled,
                y_train,
                cv=5,
                scoring="accuracy",
                n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 5),
            )

            course_performance[clf_name] = {
                "best_params": best_params,  # bets params found after hyperparams optimization
                "performance": metrics,  # auc, f1, acc
                "y_pred": y_pred,  # predictions
                "y_test": y_test,  # true labels
                "learning_curve": {  # learning curve data
                    "train_sizes": train_sizes,
                    "train_scores": train_scores,
                    "test_scores": test_scores,
                },
            }

        overall_performance[course_name] = course_performance

    print("\nOverall Classifier Performance:")
    for course_name, performance in overall_performance.items():
        print(f"\n{course_name}")
        for clf_name, clf_performance in performance.items():
            print(
                f"{clf_name} with {clf_performance['best_params']}: {clf_performance['performance']}"
            )

    return overall_performance
