# Machine Learning Assignment 2 — Classification Model Comparison

## a. Problem statement
Implement five classification algorithms on the same dataset, compare them using six evaluation metrics, and provide an interactive Streamlit application that evaluates saved models on an uploaded labelled test CSV without retraining.

## b. Dataset description
- **Dataset:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** UCI Machine Learning Repository; a copy is bundled with `scikit-learn`'s `load_breast_cancer` dataset loader.
- **Task:** Binary classification of breast-cancer samples.
- **Instances:** 569
- **Input features:** 30 numeric features
- **Target column:** `target` (`0 = malignant`, `1 = benign`)
- **Split:** stratified 75% training / 25% test, `random_state=42`.
- **Test file:** `test_data.csv` (143 rows), includes all features and the target label.
- **Preprocessing:** StandardScaler is included in pipelines for Logistic Regression, KNN, Gaussian Naive Bayes and SVM. Decision Tree is trained without feature scaling.
- **Caution:** This is a benchmark dataset, not a clinical diagnostic tool.

## c. GitHub Repository Link
Replace this placeholder after publishing the repository:
`https://github.com/<your-github-username>/<your-repository-name>`

## d. Models used and evaluation

All models were evaluated on the same stratified held-out test set. AUC uses predicted probability for class 1 (benign).

| ML Model Name          |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression    |     0.986  | 0.9977 |      0.9889 |   0.9889 | 0.9889 | 0.97   |
| Decision Tree          |     0.9161 | 0.9488 |      0.9535 |   0.9111 | 0.9318 | 0.8243 |
| KNN                    |     0.979  | 0.9845 |      0.9677 |   1      | 0.9836 | 0.9555 |
| Naive Bayes            |     0.9371 | 0.9878 |      0.9355 |   0.9667 | 0.9508 | 0.8644 |
| Support Vector Machine |     0.979  | 0.9969 |      0.9888 |   0.9778 | 0.9832 | 0.9553 |

### Observations about model performance
- **Logistic Regression:** Achieved the highest accuracy on this held-out split; achieved the highest auc; achieved the highest f1 score; achieved the highest mcc.\n- **Decision Tree:** Provided the performance shown in the comparison table; assess the precision/recall trade-off alongside accuracy.\n- **KNN:** Provided the performance shown in the comparison table; assess the precision/recall trade-off alongside accuracy.\n- **Naive Bayes:** Provided the performance shown in the comparison table; assess the precision/recall trade-off alongside accuracy.\n- **Support Vector Machine:** Provided the performance shown in the comparison table; assess the precision/recall trade-off alongside accuracy.\n
- **Overall winner on this split:** **Logistic Regression** by AUC. Compare the full metric table before making a final selection; the best model depends on the metric and the cost of false positives versus false negatives.
- These observations are specific to this split and may change with a different split or hyperparameters.

## Streamlit app
The app supports:
1. CSV upload
2. Model-selection dropdown for all five saved models
3. Accuracy, AUC, Precision, Recall, F1 and MCC
4. Confusion matrix and classification report
5. Downloadable predictions CSV

The app loads serialized models from `model/` and does not train or fine-tune models on uploaded test data.

## Run locally
```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

To regenerate model artifacts and the test data:
```bash
python train_models.py
```

## Repository contents
- `app.py` — Streamlit application
- `train_models.py` — reproducible training and evaluation script
- `requirements.txt` — Python dependencies
- `test_data.csv` — held-out test features and target
- `breast_cancer_dataset.csv` — full dataset copy for transparency/reproducibility
- `model/` — five serialized model pipelines/estimators
- `model_comparison_metrics.csv` — computed comparison table

## Deployment and submission checklist
1. Upload this project to your own GitHub repository and commit your own changes.
2. In Streamlit Community Cloud, create a new app from that repository and select `app.py`.
3. Open the deployed URL and test the CSV upload and all five model options.
4. Run the project on BITS Virtual Lab and capture the required screenshot yourself.
5. Replace the GitHub placeholder above and add the live Streamlit URL.
6. Include the complete README content in the single PDF submission as required by the assignment.

**Academic integrity:** Use this as a learning scaffold. Understand the implementation, personalize the analysis and UI, verify results in your required BITS Virtual Lab environment, and follow your course's rules for AI-assisted work.
