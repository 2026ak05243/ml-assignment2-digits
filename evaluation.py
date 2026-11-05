"""Metric helpers shared by training and the Streamlit app.

The uploaded test file is scored with the already fitted model. Nothing here
fits or updates a classifier.
"""

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def _average_kwargs(labels):
    """Binary metrics use the positive class. Multi-class metrics are weighted."""
    labels = list(labels)
    if len(labels) == 2:
        return {"average": "binary", "pos_label": labels[1], "zero_division": 0}
    return {"average": "weighted", "zero_division": 0}


def auc_score(model, X, y):
    """Weighted one-vs-rest AUC from predicted class probabilities."""
    labels = list(model.classes_)
    scores = model.predict_proba(X)
    if len(labels) == 2:
        positive_index = labels.index(labels[1])
        return float(roc_auc_score(y, scores[:, positive_index]))
    return float(
        roc_auc_score(y, scores, multi_class="ovr", average="weighted")
    )


def compute_metrics(model, X, y):
    """Return the six required metrics and the predicted labels."""
    y_pred = model.predict(X)
    labels = list(model.classes_)
    kw = _average_kwargs(labels)
    metrics = {
        "Accuracy": float(accuracy_score(y, y_pred)),
        "AUC": auc_score(model, X, y),
        "Precision": float(precision_score(y, y_pred, **kw)),
        "Recall": float(recall_score(y, y_pred, **kw)),
        "F1": float(f1_score(y, y_pred, **kw)),
        "MCC": float(matthews_corrcoef(y, y_pred)),
    }
    return metrics, y_pred


def top_confusions(y_true, y_pred, labels, limit=5):
    """Largest off-diagonal cells: (count, actual, predicted)."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    pairs = []
    for i, actual in enumerate(labels):
        for j, predicted in enumerate(labels):
            count = int(matrix[i, j])
            if i != j and count > 0:
                pairs.append((count, actual, predicted))
    pairs.sort(key=lambda item: item[0], reverse=True)
    return pairs[:limit]


def report_text(y_true, y_pred, labels):
    return classification_report(
        y_true, y_pred, labels=labels, digits=4, zero_division=0
    )
