"""Streamlit app for evaluating saved scikit-learn classifier models on a labelled CSV.

This script loads one of the pretrained models stored in the `model/` directory,
validates the uploaded dataset schema, computes standard classification metrics,
and displays the confusion matrix and classification report for the chosen model.

Usage:
    streamlit run app.py
"""

from pathlib import Path

# Core libraries required for loading models, reading CSV files, and building the UI.
# `joblib` is used because the trained scikit-learn models were serialized with it.
import joblib
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

# The app must be started through Streamlit.
if get_script_run_ctx() is None:
    raise SystemExit("This app must be run with 'streamlit run app.py'.")

# Project paths: the script sits in the app root, pretrained models are stored in the
# dedicated `model/` directory, and the generated sample/test CSVs live in `data/`.
# The target column is fixed for this dataset and is used for supervised evaluation.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR = BASE_DIR / "model"
TARGET_COLUMN = "target"

# Mapping between human-friendly model names shown in the UI and the serialized
# model files stored on disk. This makes it easy to switch models without changing
# the rest of the logic.
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

# Configure the app page title, icon, and layout so the interface is polished and
# consistent with a dashboard-style ML comparison tool.
st.set_page_config(
    page_title="Breast Cancer Model Comparison",
    page_icon="📊",
    layout="wide",
)
st.title("Breast Cancer Diagnostic Classifier Comparison")
st.caption("Compare five trained classification models on a labelled test dataset.")

with st.sidebar:
    st.header("Evaluation settings")
    # Users choose which pretrained model to evaluate. The selected option is used later
    # to locate the matching serialized model file in the `model/` directory.
    selected_model = st.selectbox("Choose a classifier", list(MODEL_FILES.keys()))
    # The uploaded CSV should contain the same feature columns as the breast cancer dataset
    # and a true label column named `target` so that predictions can be compared to ground truth.
    uploaded_file = st.file_uploader("Upload test data (CSV)", type=["csv"])
    st.info("CSV must contain all 30 dataset features and a target column named `target`. Do not upload training data.")

st.markdown(
    """
    This application evaluates pretrained models.
    Dataset: Breast Cancer Wisconsin (Diagnostic), a binary classification dataset with 30 numeric features.
    The five classifiers were trained on the same training split and saved to disk. 
    This app loads one of the models, validates the uploaded CSV schema, computes standard classification metrics, 
    and displays the confusion matrix and classification report.
    """
)

@st.cache_resource
def load_model(model_path):
    """Load a serialized scikit-learn model from disk and cache it for efficiency.

    The cache ensures that the selected model is not reloaded repeatedly on each rerun,
    which is especially useful in Streamlit where the app reruns whenever a widget changes.
    """
    return joblib.load(model_path)

# If no file is uploaded, the user can still try the repository-provided sample test CSV.
# This is helpful for demonstration purposes and ensures the app can be explored without
# manually preparing a new dataset.
if uploaded_file is None:
    st.subheader("Use Sample Test Dataset")
    default_path = DATA_DIR / "test_data.csv"
    if default_path.exists():
        st.download_button(
            "Download sample test_data.csv",
            data=default_path.read_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
        )
    st.write("Upload `test_data.csv` from the repository to display evaluation metrics.")
    st.stop()

# Read the uploaded CSV and stop gracefully if the file is invalid or unreadable.
try:
    test_df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

# A valid evaluation dataset must include the true class labels under the `target` column.
# Without that column, there is no supervised label set to compare against predictions.
if TARGET_COLUMN not in test_df.columns:
    st.error("The uploaded CSV must include a `target` column containing the true class labels.")
    st.stop()

# Locate the selected model file in the `model/` directory. This step prevents silent
# failures if the expected artifact is missing from the repository or deployment.
model_path = MODEL_DIR / MODEL_FILES[selected_model]
if not model_path.exists():
    st.error(f"Saved model file is missing: {model_path.name}")
    st.stop()

# Load the trained model and determine the exact input feature names it was trained with.
# Some scikit-learn objects expose `feature_names_in_`, which is the most reliable source.
# If that attribute is missing, a fallback uses the repository's sample CSV schema.
model = load_model(model_path)
expected_features = list(getattr(model, "feature_names_in_", []))
if not expected_features:
    # Pipelines and estimators trained with DataFrames usually expose feature_names_in_.
    expected_features = list(pd.read_csv(DATA_DIR / "test_data.csv", nrows=1).drop(columns=[TARGET_COLUMN]).columns)

# This schema check makes sure the uploaded CSV uses the same columns as the model expects.
# Extra columns are ignored, while missing required feature columns stop the evaluation.
missing = [c for c in expected_features if c not in test_df.columns]
extra = [c for c in test_df.columns if c not in expected_features and c != TARGET_COLUMN]
if missing:
    st.error(f"Missing required feature columns ({len(missing)}): {', '.join(missing)}")
    st.stop()
if extra:
    st.warning(f"Extra columns will be ignored: {', '.join(extra)}")

# Restrict the dataset to the model's exact feature list and keep the target label separate.
# This is essential because scikit-learn models expect arrays matching the training schema.
X_test = test_df[expected_features].copy()
y_true = test_df[TARGET_COLUMN].copy()

# Before computing metrics, validate the data quality. Missing feature values, null labels,
# and non-binary target values would make the evaluation invalid or unstable.
if X_test.isnull().any().any():
    st.error("The uploaded feature columns contain missing values. Please clean the CSV and upload it again.")
    st.stop()
if not set(pd.Series(y_true).dropna().unique()).issubset({0, 1}):
    st.error("The `target` column must contain only 0 and 1 for this dataset.")
    st.stop()
if y_true.isnull().any():
    st.error("The target column contains missing values.")
    st.stop()

# Run inference and compute the metric dictionary. We compute probabilities for the positive
# class so that AUC can be measured with the same decision threshold convention used in ML.
try:
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC Score": roc_auc_score(y_true, y_score) if len(set(y_true)) == 2 else float("nan"),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "MCC Score": matthews_corrcoef(y_true, y_pred),
    }
except Exception as exc:
    st.error(f"Evaluation failed. Verify that the CSV matches the expected feature schema. Details: {exc}")
    st.stop()

# The metrics are displayed as Streamlit cards so the user can quickly compare models.
st.subheader(f"Results: {selected_model}")
st.write(f"Rows evaluated: **{len(test_df)}**")
cols = st.columns(6)
for col, (label, value) in zip(cols, metrics.items()):
    col.metric(label, f"{value:.4f}" if pd.notna(value) else "N/A")

# Confusion matrix and classification report help interpret model behavior by showing how
# often the classifier confuses malignant and benign cases.
left, right = st.columns(2)
with left:
    st.markdown("#### Confusion matrix")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots()
    image = ax.imshow(cm)
    ax.set_xticks([0, 1], labels=["Predicted 0", "Predicted 1"])
    ax.set_yticks([0, 1], labels=["Actual 0", "Actual 1"])
    for (i, j), val in __import__("numpy").ndenumerate(cm):
        ax.text(j, i, str(val), ha="center", va="center")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.markdown("#### Classification report")
    report = classification_report(
        y_true, y_pred, labels=[0, 1],
        target_names=["malignant (0)", "benign (1)"],
        output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

# Exporting predictions is useful for auditability and for users who want to inspect a CSV
# containing both the selected features and the model's class assignments.
st.download_button(
    "Download predictions CSV",
    data=pd.concat([X_test.reset_index(drop=True),
                    y_true.reset_index(drop=True).rename("actual_target"),
                    pd.Series(y_pred, name="predicted_target")], axis=1).to_csv(index=False).encode("utf-8"),
    file_name=f"{selected_model.lower().replace(' ', '_')}_predictions.csv",
    mime="text/csv"
)
