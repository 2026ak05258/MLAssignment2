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

# Keep path definitions free of side effects; only main() writes output files.
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"


def main():
    """Train five classifiers, evaluate the held-out split, and export artifacts.

    Takes no arguments and returns None. Creates model/ and data/ if needed,
    writes one joblib artifact per model, and exports test_data.csv,
    breast_cancer_dataset.csv, and model_comparison_metrics.csv under data/.
    The comparison table is also printed to standard output.

    Existing output files are overwritten. Training, evaluation, and file I/O
    errors propagate to the caller; earlier writes are not rolled back if a
    later operation fails. Retraining is not required to launch the app when
    compatible saved artifacts are already available.
    """
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    # Load the clean, labelled dataset with its original feature names.
    X, y = load_breast_cancer(as_frame=True, return_X_y=True)
    # Use the same 426 training / 143 test samples for every model. Stratification
    # preserves the class balance; a fixed seed makes the split reproducible.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    # Scaling is fitted on training data only, as part of each pipeline, preventing
    # test-data leakage. The decision tree uses the unscaled numeric features.
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

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        # Class 1 is benign; locate its probability column explicitly.
        positive_index = list(model.classes_).index(1)
        score = model.predict_proba(X_test)[:, positive_index]
        rows.append({
            "ML Model Name": name,
            "Accuracy": accuracy_score(y_test, pred),
            "AUC": roc_auc_score(y_test, score),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "MCC": matthews_corrcoef(y_test, pred),
        })

        # Keep these filenames consistent with MODEL_FILES in app.py. Serializing
        # the whole pipeline retains its fitted preprocessing for later inference.
        output_name = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, MODEL_DIR / output_name)

    # Preserve index alignment between the features and their true labels.
    test_df = X_test.copy()
    test_df["target"] = y_test
    test_df.to_csv(DATA_DIR / "test_data.csv", index=False)

    # The full dataset includes training rows: this export is for reference, not
    # an independent test set for measuring generalization performance.
    full_df = X.copy()
    full_df["target"] = y
    full_df.to_csv(DATA_DIR / "breast_cancer_dataset.csv", index=False)

    # Export unrounded metric values; round only the human-readable console view.
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(DATA_DIR / "model_comparison_metrics.csv", index=False)
    print(metrics_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
