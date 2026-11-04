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
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
)

# The app must be started through Streamlit.
if get_script_run_ctx() is None:
    raise SystemExit("This app must be run with 'streamlit run app.py'.")

# Project paths: the script sits in the app root, pretrained models are stored in the
# dedicated `model/` directory, and the generated sample/test CSVs live in `data/`.
# The target column is fixed for this dataset and is used for supervised evaluation.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
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

st.markdown(
    """
    <style>
    .stApp { background-color: #f3f3f3; }
    .block-container { padding-top: 1.5rem; }
    [data-testid="stSidebar"] {
        background-color: #f1f1f1;
        border-right: 1px solid #d9d9d9;
    }
    .sidebar-card {
        background: #edf3f0;
        border: 1px solid #d5e5dc;
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        margin-bottom: 1rem;
        color: #204c39;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .sidebar-section {
        margin-top: 1rem;
        color: #2c2c2c;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .sidebar-note {
        font-size: 0.82rem;
        color: #5f5f5f;
        margin-top: 0.75rem;
        line-height: 1.4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

use_sample_data = False

with st.sidebar:
    st.markdown("<div style='font-size:1.1rem; font-weight:600; color:#2b2b2b; margin-bottom:0.5rem;'>Breast Cancer</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:1.05rem; color:#2b2b2b; margin-bottom:0.8rem;'>Evaluation</div>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-card">5 pre-trained models<br>for comparison</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">1. Model</div>', unsafe_allow_html=True)
    # All available models are evaluated; the selection controls the detailed
    # metrics, confusion matrix, classification report, and prediction download.
    selected_model = st.selectbox("Select model for detailed view", list(MODEL_FILES.keys()), index=0)

    st.markdown('<div class="sidebar-section">2. Test data</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload test CSV", type=["csv"])

    # Keep the bundled sample option visible and aligned with the checkbox in a single row,
    # with the checkbox appearing before the descriptive text.
    checkbox_col, label_col = st.columns([1, 7])
    use_sample_data = checkbox_col.checkbox("Use bundled test_data.csv", value=False, label_visibility="collapsed")
    label_col.markdown(
        "<div style='padding-top: 0.45rem; font-weight: 600; color: #1d4f2d;'>Use bundled test_data.csv</div>",
        unsafe_allow_html=True,
    )


st.title("Breast Cancer Model Evaluation")

st.markdown(
    """
    This application evaluates pretrained models.
    Dataset: Breast Cancer Wisconsin (Diagnostic), a binary classification dataset with 30 numeric features.
    The five classifiers were trained on the same training split and saved to disk.
    This app loads one of the models, validates the uploaded CSV schema, computes standard classification metrics,
    and displays the confusion matrix and classification report.
    """
)

# Stop before model loading or CSV reading until a data source is explicitly chosen.
show_tabs = use_sample_data or uploaded_file is not None
if not show_tabs:
    st.info('Upload a test CSV from the sidebar, or tick "Use bundled test_data.csv".')
    st.info("CSV must contain all 30 dataset features and a target column named `target`. Do not upload training data.")
    st.stop()

st.caption("Educational benchmark only, not a clinical diagnostic tool. Positive class: benign (1); malignant: 0.")


@st.cache_resource
def load_model(model_path):
    """Load a trusted local estimator and reuse it across Streamlit reruns.

    Args:
        model_path: Path to a joblib artifact from the project's model directory.

    Returns:
        A fitted estimator or pipeline exposing predict, predict_proba, and classes_.

    Notes:
        Loading errors propagate to the per-model error handler. Cached models
        are shared resources and must not be fitted or mutated here. Replacing
        an artifact at the same path does not invalidate this cache; clear the
        resource cache or restart the app after regenerating models.
    """
    return joblib.load(model_path)


def evaluate_model(model, test_df, y_true):
    """Evaluate one fitted binary classifier without modifying inputs or training.

    Args:
        model: Fitted classifier or pipeline with classes 0 and 1 and probability
            prediction support. Feature names are read from feature_names_in_.
        test_df: Nonempty DataFrame containing the model's required feature
            columns. Extra columns are excluded from inference.
        y_true: Validated, non-null binary label Series in the same row order
            and with the same length as test_df; checked by the caller.

    Returns:
        Tuple of (metric dictionary, predicted labels, benign-class probability
        array, ignored column names). Predictions preserve input row order.
        AUC is NaN when only one true class is present. Undefined precision,
        recall, or F1 uses zero_division=0.

    Raises:
        ValueError: Required features are missing, values are nonnumeric or
            nonfinite, the schema is empty, or model classes are not 0 and 1.
        Other schema-loading and inference errors propagate to the caller's
        per-model handler, which displays the failure in the app.
    """
    required = list(getattr(model, "feature_names_in_", []))
    if not required:
        # Legacy artifacts without feature metadata rely on the bundled CSV's
        # header; never infer the training schema from an arbitrary upload.
        required = list(
            pd.read_csv(DATA_DIR / "test_data.csv", nrows=0)
            .drop(columns=[TARGET_COLUMN]).columns
        )
    if not required:
        raise ValueError("Could not determine the model's input features.")
    missing = [column for column in required if column not in test_df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {', '.join(missing)}")

    # Reorder columns to the training schema; fitted pipelines perform scaling.
    features = test_df[required].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Feature columns must contain finite numeric values, without missing values.")

    classes = list(model.classes_)
    if len(classes) != 2 or set(classes) != {0, 1}:
        raise ValueError("The saved model must classify malignant (0) and benign (1) samples.")
    predictions = model.predict(features)
    # Probability columns follow classes_, not an assumed fixed label order.
    scores = model.predict_proba(features)[:, classes.index(1)]
    result = {
        "Accuracy": accuracy_score(y_true, predictions),
        "AUC": roc_auc_score(y_true, scores) if y_true.nunique() == 2 else float("nan"),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1 Score": f1_score(y_true, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y_true, predictions),
    }
    extra = [column for column in test_df.columns if column not in required and column != TARGET_COLUMN]
    return result, predictions, scores, extra


# The bundled-data checkbox takes precedence if an upload is also present.
# Read only the explicitly selected source and handle errors for either CSV source.
try:
    if use_sample_data:
        if uploaded_file is not None:
            st.info("Bundled test data is selected; the uploaded CSV is not being used.")
        test_df = pd.read_csv(DATA_DIR / "test_data.csv")
    else:
        test_df = pd.read_csv(uploaded_file)
except (OSError, ValueError) as exc:
    st.error(f"Could not read the test CSV: {exc}")
    st.stop()

# Validate labels once for all models; feature validation remains model-specific.
if test_df.empty:
    st.error("The test CSV must contain at least one labelled sample.")
    st.stop()
if TARGET_COLUMN not in test_df.columns:
    st.error("The test CSV must include a `target` column containing the true class labels.")
    st.stop()
try:
    y_true = pd.to_numeric(test_df[TARGET_COLUMN], errors="raise")
except (TypeError, ValueError):
    st.error("The `target` column must contain only 0 and 1.")
    st.stop()
if y_true.isna().any() or not y_true.isin([0, 1]).all():
    st.error("The `target` column must contain only 0 and 1, with no missing values.")
    st.stop()
y_true = y_true.astype(int)
if y_true.nunique() < 2:
    st.warning("Only one target class is present. AUC is unavailable; other metrics may be misleading.")

# Evaluate each model once. An unavailable comparison model must not crash the page.
model_metrics = {}
selected_result = None
for model_name, filename in MODEL_FILES.items():
    try:
        model = load_model(MODEL_DIR / filename)
        result = evaluate_model(model, test_df, y_true)
    except Exception as exc:
        st.warning(f"Could not evaluate {model_name}: {exc}")
        continue
    model_metrics[model_name] = result[0]
    if model_name == selected_model:
        # Reuse these outputs for details and export instead of predicting twice.
        selected_result = result

if not model_metrics:
    st.error("No models could be evaluated. Check the saved models and the test CSV.")
    st.stop()
if selected_result is None:
    # Do not silently substitute another model in the selected model's report.
    st.error("The selected model could not be evaluated. Choose another model or correct the issue above.")
    st.stop()

metrics, y_pred, y_score, extra = selected_result
if extra:
    st.warning(f"Extra columns ignored by the selected model: {', '.join(extra)}")
if len(model_metrics) < len(MODEL_FILES):
    st.caption("Comparison and best-model summaries include only successfully evaluated models.")

comparison_df = pd.DataFrame(model_metrics).T.reset_index().rename(columns={"index": "Model"})
metric_columns = ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]

# Tab layout matching the reference dashboard structure.
if show_tabs:
    evaluation_tab, confusion_tab = st.tabs([
        "Evaluation Metrics",
        "Confusion Matrix & Report",
    ])

    with evaluation_tab:
        st.markdown("### Evaluation metrics — all models on this test data")
        display_df = comparison_df[["Model"] + metric_columns].copy()
        display_df.index = range(1, len(display_df) + 1)
        # An all-NaN AUC column cannot be meaningfully color-scaled.
        gradient_columns = [col for col in metric_columns if display_df[col].notna().any()]
        st.dataframe(
            display_df.style.format({col: "{:.4f}" for col in metric_columns}, na_rep="N/A")
            .background_gradient(cmap="RdYlGn", axis=0, subset=gradient_columns),
            use_container_width=True,
        )

        # Winners refer only to successful evaluations on this CSV, not a
        # universal ranking. The baseline always predicts its most common label.
        best_model_row = comparison_df.sort_values("MCC", ascending=False).iloc[0]
        best_accuracy_row = comparison_df.sort_values("Accuracy", ascending=False).iloc[0]
        majority_baseline = y_true.value_counts(normalize=True).max()

        metric_cols = st.columns(3)
        metric_cols[0].markdown("**Best model (MCC)**")
        metric_cols[0].markdown(
            f"<div style='font-size:2.1rem; font-weight:500; line-height:1.2;'>{best_model_row['Model']}</div>",
            unsafe_allow_html=True,
        )
        metric_cols[0].markdown(f"<span style='color:#2e7d32; font-size:1.2rem;'>↑ {best_model_row['MCC']:.4f}</span>", unsafe_allow_html=True)

        metric_cols[1].markdown("**Best accuracy**")
        metric_cols[1].markdown(
            f"<div style='font-size:2.1rem; font-weight:500; line-height:1.2;'>{best_accuracy_row['Model']}</div>",
            unsafe_allow_html=True,
        )
        metric_cols[1].markdown(f"<span style='color:#2e7d32; font-size:1.2rem;'>↑ {best_accuracy_row['Accuracy']:.4f}</span>", unsafe_allow_html=True)

        metric_cols[2].markdown("**Majority-class baseline**")
        metric_cols[2].markdown(f"<div style='font-size:2.1rem; font-weight:500; line-height:1.2;'>{majority_baseline:.4f}</div>", unsafe_allow_html=True)

    with confusion_tab:
        # Confusion matrix and classification report help interpret model behavior by showing how
        # often the classifier confuses malignant and benign cases.
        st.subheader(f"Results: {selected_model}")
        st.write(f"Rows evaluated: **{len(test_df)}**")
        cols = st.columns(6)
        for col, (label, value) in zip(cols, metrics.items()):
            col.metric(label, f"{value:.4f}" if pd.notna(value) else "N/A")

        left, right = st.columns(2)
        with left:
            st.markdown("#### Confusion matrix")
            # Explicit labels retain a 2x2 matrix even for single-class test data.
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            fig, ax = plt.subplots()
            ax.imshow(cm)
            ax.set_xticks([0, 1], labels=["Predicted 0", "Predicted 1"])
            ax.set_yticks([0, 1], labels=["Actual 0", "Actual 1"])
            for (i, j), val in np.ndenumerate(cm):
                ax.text(j, i, str(val), ha="center", va="center")
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("True label")
            fig.tight_layout()
            st.pyplot(fig)
            # Streamlit reruns frequently; release the figure after rendering.
            plt.close(fig)
        with right:
            st.markdown("#### Classification report")
            report = classification_report(
                y_true, y_pred, labels=[0, 1],
                target_names=["malignant (0)", "benign (1)"],
                output_dict=True, zero_division=0
            )
            st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

        # Retain input columns and row order. These two reserved output columns
        # replace same-named upload columns, if present; no server file is written.
        predictions_df = test_df.copy()
        predictions_df["predicted_target"] = y_pred
        predictions_df["probability_benign"] = y_score
        st.download_button(
            "Download predictions CSV",
            data=predictions_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{Path(MODEL_FILES[selected_model]).stem}_predictions.csv",
            mime="text/csv",
        )



