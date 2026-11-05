"""Streamlit front end for the five saved digit classifiers.

The app loads the joblib pipelines, scores an uploaded test CSV, and does not
retrain. Launch it from this folder with: streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from evaluation import METRIC_ORDER, compute_metrics

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
SCHEMA_PATH = ROOT / "artifacts" / "schema.json"
BUNDLED_TEST = ROOT / "test_data.csv"


@st.cache_resource
def load_bundle():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    models = {}
    for name, filename in schema["models"].items():
        models[name] = joblib.load(MODEL_DIR / filename)
    return schema, models


def read_test_frame(uploaded_file, schema):
    if uploaded_file is None:
        frame = pd.read_csv(BUNDLED_TEST)
        source = "bundled test_data.csv (the held-out 20 percent split)"
    else:
        frame = pd.read_csv(uploaded_file)
        source = uploaded_file.name

    target = schema["target"]
    features = schema["features"]
    missing = [column for column in features + [target] if column not in frame.columns]
    if missing:
        st.error(
            "This CSV is missing required columns: "
            + ", ".join(missing)
            + ". Keep all 64 pixel columns and the digit column."
        )
        st.stop()

    features_frame = frame[features].apply(pd.to_numeric, errors="coerce")
    target_series = pd.to_numeric(frame[target], errors="coerce")
    if features_frame.isna().any().any() or target_series.isna().any():
        st.error("Feature values and the digit column must be numeric.")
        st.stop()
    return features_frame, target_series.astype(int), source


def metric_frame(metrics):
    return pd.DataFrame(
        [{"Metric": name, "Score": round(metrics[name], 4)} for name in METRIC_ORDER]
    )


def main():
    st.set_page_config(page_title="Optical Digit Classifiers", layout="wide")
    st.title("Handwritten digit classification")
    st.write(
        "Five saved models score the UCI optical-digits test file. "
        "Pixel intensities stay in the 8 by 8 layout named pixel_row_column. "
        "The target column is digit. Models are loaded from disk and are not retrained."
    )

    if not SCHEMA_PATH.exists() or not BUNDLED_TEST.exists():
        st.error("Run python train_models.py in this folder before starting the app.")
        st.stop()

    schema, models = load_bundle()
    model_names = list(schema["models"].keys())

    with st.sidebar:
        st.header("Evaluation setup")
        uploaded = st.file_uploader(
            "Upload a test CSV",
            type=["csv"],
            help="Include the 64 pixel features and the digit column. Upload the test split only.",
        )
        selected = st.selectbox("Model", model_names, index=model_names.index("Support Vector Machine") if "Support Vector Machine" in model_names else 0)
        st.caption("Leave the uploader empty to score the included test_data.csv.")

    X, y, source = read_test_frame(uploaded, schema)
    st.info(f"Scoring {len(X)} rows from {source}.")

    sample_columns = st.columns(5)
    for column, position in zip(sample_columns, range(min(5, len(X)))):
        image = X.iloc[position].to_numpy().reshape(8, 8)
        figure, axis = plt.subplots(figsize=(2.2, 2.2))
        axis.imshow(image, cmap="gray_r", vmin=0, vmax=16)
        axis.set_title(f"digit {int(y.iloc[position])}")
        axis.axis("off")
        column.pyplot(figure)
        plt.close(figure)

    selected_model = models[selected]
    selected_metrics, y_pred = compute_metrics(selected_model, X, y)
    labels = list(selected_model.classes_)

    st.subheader(selected)
    st.dataframe(metric_frame(selected_metrics), hide_index=True, use_container_width=True)

    matrix_column, report_column = st.columns(2)
    with matrix_column:
        st.markdown("**Confusion matrix**")
        matrix = confusion_matrix(y, y_pred, labels=labels)
        figure, axis = plt.subplots(figsize=(6.5, 5))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=axis,
        )
        axis.set_xlabel("Predicted digit")
        axis.set_ylabel("Actual digit")
        st.pyplot(figure)
        plt.close(figure)

    with report_column:
        st.markdown("**Classification report**")
        report = classification_report(y, y_pred, labels=labels, digits=4, zero_division=0)
        st.code(report, language="text")

    predicted = pd.Series(y_pred, index=X.index)
    mistakes = X.loc[predicted.to_numpy() != y.to_numpy()].head(5)
    if mistakes.empty:
        st.success("The selected model classified every row in this file correctly.")
    else:
        st.markdown("**A few misclassified digits**")
        mistake_columns = st.columns(len(mistakes))
        for column, (index, _) in zip(mistake_columns, mistakes.iterrows()):
            image = X.loc[index].to_numpy().reshape(8, 8)
            figure, axis = plt.subplots(figsize=(2.2, 2.2))
            axis.imshow(image, cmap="gray_r", vmin=0, vmax=16)
            axis.set_title(f"true {int(y.loc[index])} / pred {int(predicted.loc[index])}")
            axis.axis("off")
            column.pyplot(figure)
            plt.close(figure)

    st.subheader("All five models on this file")
    rows = []
    for name, model in models.items():
        metrics, _ = compute_metrics(model, X, y)
        row = {"ML Model Name": name}
        row.update({metric: round(metrics[metric], 4) for metric in METRIC_ORDER})
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


main()
