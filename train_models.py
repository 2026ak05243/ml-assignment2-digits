"""Train the five required classifiers on the UCI optical digits dataset.

Run from this folder:

    python train_models.py

The script writes model/*.joblib, test_data.csv, and artifacts/summary.json.
It does not download anything. scikit-learn already ships this UCI dataset.
"""

from __future__ import annotations

import json
import platform
import warnings
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from evaluation import METRIC_ORDER, compute_metrics, report_text, top_confusions

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
ARTIFACT_DIR = ROOT / "artifacts"
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "digit"

# Dropdown labels stay in the order required by the assignment.
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def load_frame():
    bunch = load_digits(as_frame=True)
    frame = bunch.frame.copy()
    frame = frame.rename(columns={"target": TARGET})
    features = [column for column in frame.columns if column != TARGET]
    return frame, features


def build_search_spaces():
    """Small grids. Selection uses training data only, through stratified CV."""
    logistic = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=4000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    tree = DecisionTreeClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    knn = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier()),
        ]
    )
    naive_bayes = MultinomialNB()
    svm = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return [
        (
            "Logistic Regression",
            logistic,
            {"clf__C": [0.1, 1.0, 10.0]},
            5,
        ),
        (
            "Decision Tree",
            tree,
            {"max_depth": [8, 16, None], "min_samples_leaf": [1, 3]},
            5,
        ),
        (
            "KNN",
            knn,
            {"clf__n_neighbors": [3, 5, 9], "clf__weights": ["uniform", "distance"]},
            5,
        ),
        (
            "Naive Bayes",
            naive_bayes,
            {"alpha": [0.1, 0.5, 1.0]},
            5,
        ),
        (
            "Support Vector Machine",
            svm,
            {"clf__C": [1.0, 10.0], "clf__gamma": ["scale"]},
            3,
        ),
    ]


def fit_model(name, estimator, grid, cv, X_train, y_train):
    search = GridSearchCV(
        estimator,
        grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    # Probability estimates are required for AUC. They are fit only for the
    # chosen SVM, after the accuracy search, so the search itself stays fast.
    if name == "Support Vector Machine":
        params = search.best_params_
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        C=params["clf__C"],
                        gamma=params["clf__gamma"],
                        probability=True,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The `probability` parameter was deprecated",
                category=FutureWarning,
            )
            model.fit(X_train, y_train)
    return model, search.best_params_, float(search.best_score_)


def choose_winner(comparison):
    """Highest F1. A gap under 0.001 is treated as a tie and broken by AUC, then MCC."""
    best_f1 = max(metrics["F1"] for _, metrics in comparison)
    contenders = [
        (name, metrics)
        for name, metrics in comparison
        if best_f1 - metrics["F1"] < 0.001
    ]
    contenders.sort(
        key=lambda item: (item[1]["AUC"], item[1]["MCC"], item[1]["F1"]),
        reverse=True,
    )
    return contenders[0][0]


def format_table(rows):
    headers = ["ML Model Name", *METRIC_ORDER]
    lines = [
        f"{'ML Model Name':<28} {'Accuracy':>10} {'AUC':>10} "
        f"{'Precision':>10} {'Recall':>10} {'F1':>10} {'MCC':>10}"
    ]
    lines.append("-" * len(lines[0]))
    for name, metrics in rows:
        lines.append(
            f"{name:<28} "
            + " ".join(f"{metrics[key]:>10.4f}" for key in METRIC_ORDER)
        )
    return "\n".join(lines)


def main():
    print("Python", platform.python_version())
    print("scikit-learn", sklearn.__version__)
    print("pandas", pd.__version__)

    frame, features = load_frame()
    X = frame[features]
    y = frame[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    test_frame = X_test.copy()
    test_frame[TARGET] = y_test.to_numpy()
    test_frame.to_csv(ROOT / "test_data.csv", index=False)

    summary = {
        "dataset": "UCI Optical Recognition of Handwritten Digits (sklearn load_digits)",
        "n_rows": int(len(frame)),
        "n_features": int(len(features)),
        "features": features,
        "target": TARGET,
        "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "split": "stratified hold-out",
        "metric_notes": {
            "Precision": "weighted average across digits 0-9",
            "Recall": "weighted average across digits 0-9",
            "F1": "weighted average across digits 0-9",
            "AUC": "one-vs-rest, weighted by class support, from predict_proba",
            "MCC": "multiclass Matthews correlation coefficient",
        },
        "models": {},
    }

    print()
    print(f"Rows: {len(frame)} | Features: {len(features)} | Classes: 10")
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print()

    comparison = []
    report_chunks = []
    for name, estimator, grid, cv in build_search_spaces():
        print(f"Fitting {name} ...")
        model, params, cv_accuracy = fit_model(
            name, estimator, grid, cv, X_train, y_train
        )
        test_metrics, y_pred = compute_metrics(model, X_test, y_test)
        train_metrics, _ = compute_metrics(model, X_train, y_train)
        labels = list(model.classes_)
        confusions = [
            {"count": count, "actual": int(actual), "predicted": int(predicted)}
            for count, actual, predicted in top_confusions(y_test, y_pred, labels)
        ]
        joblib.dump(model, MODEL_DIR / MODEL_FILES[name], compress=3)
        summary["models"][name] = {
            "file": MODEL_FILES[name],
            "best_params": {key: _jsonify(value) for key, value in params.items()},
            "cv_accuracy": round(cv_accuracy, 4),
            "train_metrics": {key: round(value, 4) for key, value in train_metrics.items()},
            "test_metrics": {key: round(value, 4) for key, value in test_metrics.items()},
            "top_confusions": confusions,
        }
        comparison.append((name, test_metrics))
        report_chunks.append(f"===== {name} =====\n")
        report_chunks.append(report_text(y_test, y_pred, labels))
        report_chunks.append("\n")
        print(
            f"  CV accuracy {cv_accuracy:.4f} | "
            f"test F1 {test_metrics['F1']:.4f} | "
            f"params {params}"
        )

    winner = choose_winner(comparison)
    summary["winner"] = winner
    summary["winner_rule"] = (
        "Highest held-out F1. If two F1 scores differ by less than 0.001, "
        "break the tie with AUC and then MCC."
    )

    schema = {
        "target": TARGET,
        "features": features,
        "models": MODEL_FILES,
    }
    (ARTIFACT_DIR / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    (ARTIFACT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (ARTIFACT_DIR / "classification_reports.txt").write_text(
        "".join(report_chunks), encoding="utf-8"
    )

    table = format_table(comparison)
    print()
    print("Held-out test metrics")
    print(table)
    print()
    print(f"Winner: {winner}")
    print("Rule: highest test F1; if the F1 gap is under 0.001, break the tie with AUC then MCC.")
    print("Saved models in model/ and the test split in test_data.csv")
    print("Screenshot this table on BITS Virtual Lab.")


def _jsonify(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


if __name__ == "__main__":
    main()
