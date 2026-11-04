# Machine Learning Assignment 2 — Classification Model Comparison

## a. Problem statement
Develop a binary classification system for breast cancer diagnosis using the Breast Cancer Wisconsin (Diagnostic) dataset. Train and compare five machine learning classifiers on the same data, evaluate them using six standard metrics, and build an interactive Streamlit application that loads the saved models and assesses an uploaded labelled test CSV without retraining.

## b. Dataset description

| Item | Details |
|:------|:--------|
| **Dataset** | Breast Cancer Wisconsin (Diagnostic) |
| **Source** | UCI Machine Learning Repository; a copy is also available through `scikit-learn` via `load_breast_cancer` |
| **Task** | Binary classification to distinguish between malignant and benign tumors |
| **Instances** | 569 samples |
| **Input features** | 30 numeric features extracted from digitized images of fine needle aspirates (FNA) |
| **Target column** | `target` (`0 = malignant`, `1 = benign`) |
| **Split** | Stratified 75% training / 25% testing with `random_state=42` |
| **Test file** | [data/test_data.csv](data/test_data.csv) contains 143 labelled samples with all features and the target value |
| **Preprocessing** | `StandardScaler` is used in pipelines for Logistic Regression, KNN, Gaussian Naive Bayes, and SVM; Decision Tree is trained without scaling |
| **Caution** | This is a benchmark dataset for educational model comparison and not a clinical diagnostic system |

## c. GitHub Repository Link

| Resource | Link |
|:---------|:-----|
| GitHub repository | https://github.com/2026ak05258/MLAssignment2 |
| Live Streamlit app | https://mlassignment2-2026ak05258.streamlit.app/ |

### Repository contents

| File or directory | Purpose |
|:------------------|:--------|
| [README.md](README.md) | Project overview, dataset description, model evaluation results, setup instructions, and repository contents |
| [app.py](app.py) | Streamlit application |
| [train_models.py](train_models.py) | Reproducible training and evaluation script |
| [requirements.txt](requirements.txt) | Python dependencies |
| [data/test_data.csv](data/test_data.csv) | Held-out test features and target |
| [data/breast_cancer_dataset.csv](data/breast_cancer_dataset.csv) | Full dataset copy for transparency and reproducibility |
| [model/](model/) | Five serialized model pipelines/estimators |
| [model/logistic_regression.joblib](model/logistic_regression.joblib) | Logistic Regression pipeline with `StandardScaler`; training configuration: `max_iter=3000`, `random_state=42` |
| [model/decision_tree.joblib](model/decision_tree.joblib) | Decision Tree classifier without feature scaling; training configuration: `max_depth=5`, `min_samples_leaf=3`, `random_state=42` |
| [model/knn.joblib](model/knn.joblib) | K-Nearest Neighbors pipeline with `StandardScaler`; training configuration: `n_neighbors=5` |
| [model/naive_bayes.joblib](model/naive_bayes.joblib) | Gaussian Naive Bayes pipeline with `StandardScaler`; classifier uses default hyperparameters |
| [model/support_vector_machine.joblib](model/support_vector_machine.joblib) | Support Vector Machine pipeline with `StandardScaler`; training configuration: RBF kernel, `probability=True`, `random_state=42` |

## d. Models used and evaluation

All models were evaluated on the same stratified held-out test set. AUC uses predicted probability for class 1 (benign).

| ML Model Name          |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression    |     0.9860 | 0.9977 |      0.9889 |   0.9889 | 0.9889 | 0.9700 |
| Decision Tree          |     0.9161 | 0.9488 |      0.9535 |   0.9111 | 0.9318 | 0.8243 |
| KNN                    |     0.9790 | 0.9845 |      0.9677 |   1.0000 | 0.9836 | 0.9555 |
| Naive Bayes            |     0.9371 | 0.9878 |      0.9355 |   0.9667 | 0.9508 | 0.8644 |
| Support Vector Machine |     0.9790 | 0.9969 |      0.9888 |   0.9778 | 0.9832 | 0.9553 |


| Metric | Best Model | Value |
|:--------|:-----------|------:|
| Best model by MCC | Logistic Regression | 0.9700 |
| Best accuracy | Logistic Regression | 0.9860 |
| Best AUC | Logistic Regression | 0.9977 |
| Best F1 score | Logistic Regression | 0.9889 |
| Majority-class baseline | Baseline | 0.6294 |

### Model ranking by MCC

Models are ranked by MCC in descending order.

| Rank | Model | Accuracy | AUC | F1 Score | MCC |
|:-----|:------|---------:|----:|---------:|----:|
| 1 | Logistic Regression | 0.9860 | 0.9977 | 0.9889 | 0.9700 |
| 2 | KNN | 0.9790 | 0.9845 | 0.9836 | 0.9555 |
| 3 | Support Vector Machine | 0.9790 | 0.9969 | 0.9832 | 0.9553 |
| 4 | Naive Bayes | 0.9371 | 0.9878 | 0.9508 | 0.8644 |
| 5 | Decision Tree | 0.9161 | 0.9488 | 0.9318 | 0.8243 |

**Overall winner on this split:** Logistic Regression achieved the best overall performance across accuracy, AUC, F1 score, and MCC.

### Observations about model performance

| Model | Performance observations |
|:------|:-------------------------|
| Logistic Regression | Highest accuracy (0.9860), AUC (0.9977), F1 score (0.9889), and MCC (0.9700) on this held-out split. |
| Decision Tree | Lowest accuracy (0.9161), AUC (0.9488), and MCC (0.8243) among the five models on this split. |
| KNN | Highest recall (1.0000) for the positive class (benign); second-highest MCC (0.9555), with accuracy of 0.9790. |
| Naive Bayes | AUC of 0.9878 and recall of 0.9667, but lower accuracy (0.9371) and MCC (0.8644) than Logistic Regression, KNN, and SVM. |
| Support Vector Machine | Second-highest AUC (0.9969); matches KNN's accuracy (0.9790), with a slightly lower MCC (0.9553). |
| **Overall winner and justification** | **Logistic Regression** achieved the highest accuracy (0.9860), AUC (0.9977), F1 score (0.9889), and MCC (0.9700), making it the strongest overall performer among the five models on this held-out test split. |


## Streamlit app
Live Streamlit app : https://mlassignment2-2026ak05258.streamlit.app

| Feature | Description |
|:--------|:------------|
| CSV upload | Upload labelled test data for evaluation. |
| Model selection | Choose the model for the detailed report and prediction export; the comparison evaluates all available models. |
| Evaluation metrics | View Accuracy, AUC, Precision, Recall, F1 score, and MCC. |
| Detailed results | Inspect the confusion matrix and classification report. |
| Prediction export | Download predictions as a CSV file. |

The app loads serialized models from `model/` and does not train or fine-tune models on uploaded test data.

## Run locally
pip install -r requirements.txt
streamlit run app.py

To regenerate model artifacts and the test data:
python train_models.py
