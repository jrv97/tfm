import warnings

from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from config import TARGET

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)


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
            "params": {"n_estimators": [100, 200], "min_samples_leaf": [1, 2]},
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
                "max_iter": [200, 300],
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
            "model": SVC(),
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

            course_performance[clf_name] = {
                "best_params": best_params,
                "performance": metrics,
            }

        overall_performance[course_name] = course_performance

    print("\nOverall Classifier Performance:")
    for course_name, performance in overall_performance.items():
        print(f"\n{course_name}")
        for clf_name, clf_performance in performance.items():
            print(
                f"{clf_name} with {clf_performance['best_params']}: {clf_performance['performance']}"
            )
