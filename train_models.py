"""Train and serialize five classification models for the breast cancer dataset.

This script loads the built-in scikit-learn breast cancer dataset, splits it into
training and held-out test partitions, trains five classifiers, saves each model to
`model/`, and exports supporting CSV files for app use and reporting.

The resulting artifacts are used by the Streamlit app in `app.py` to compare model
performance on user-uploaded CSV files without retraining the models.

Run:
    python train_models.py
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

# Project root and model output folder. The script creates the model directory if it
# does not already exist so all serialized artifacts are written in a predictable place.
ROOT = Path(__file__).resolve().parent
(ROOT / "model").mkdir(exist_ok=True)

# Load the built-in breast cancer dataset from scikit-learn. It is already a clean,
# labelled binary classification dataset with 30 numeric features and a target column.
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target

# Create a stratified train-test split so the model evaluation is representative and
# both classes are preserved in the held-out set.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

# Define the five classifiers used in the assignment. Some use pipelines with scaling,
# while the decision tree is intentionally left unscaled to reflect a simple baseline.
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=3000, random_state=42)),
    ]),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_leaf=3),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5)),
    ]),
    "Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", GaussianNB()),
    ]),
    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(probability=True, random_state=42, kernel="rbf")),
    ]),
}

# Train each model, save it to disk, compute evaluation metrics on the held-out test set,
# and collect the results in a list that will later be exported as CSV.
rows = []
for name, model in models.items():
    # Fit the model on the training split only.
    model.fit(X_train, y_train)

    # Persist the fully trained estimator so the Streamlit app can load it later without
    # retraining or exposing the training pipeline to end users.
    output_name = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(model, ROOT / "model" / output_name)

    # Generate predictions and the positive-class probability scores needed for AUC.
    pred = model.predict(X_test)
    score = model.predict_proba(X_test)[:, 1]

    # Store all key classification metrics for later comparison and reporting.
    rows.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, score),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, pred),
    })

# Export the held-out test split as a CSV that includes the feature columns and the true
# target values. This file is used by the app as the default sample dataset for evaluation.
test_df = X_test.copy()
test_df["target"] = y_test
test_df.to_csv(ROOT / "test_data.csv", index=False)

# Export the full dataset with labels to support transparency and reproducibility.
full_df = X.copy()
full_df["target"] = y
full_df.to_csv(ROOT / "breast_cancer_dataset.csv", index=False)

# Save the metric comparison table to CSV for easy reading in Excel or GitHub review.
metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(ROOT / "model_comparison_metrics.csv", index=False)

# Print the final metrics table to the terminal for quick visual inspection during training.
print(metrics_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
