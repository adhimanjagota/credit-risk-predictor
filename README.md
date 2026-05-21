# Credit Risk Predictor

A machine learning pipeline that predicts the probability of loan default using real-world financial features. Trains and compares three classifiers, selects the best by ROC-AUC, and generates visual evaluation reports.

---

## What it does

- Generates a realistic synthetic dataset of 2,500 loan applicants with 10 financial features
- Trains and compares **Random Forest**, **Gradient Boosting**, and **Logistic Regression**
- Selects best model via 5-fold cross-validation (ROC-AUC)
- Outputs a full classification report, confusion matrix, ROC curve, and feature importance chart
- Predicts default probability for individual applicants

---

## Demo output

```
=== Credit Risk Predictor — Model Comparison ===

  Random Forest             CV AUC: 0.8821   Test AUC: 0.8904  <-- best
  Gradient Boosting         CV AUC: 0.8743   Test AUC: 0.8812
  Logistic Regression       CV AUC: 0.8201   Test AUC: 0.8187

=== Best Model: Random Forest ===

              precision    recall  f1-score
  No Default     0.91      0.94      0.92
  Default        0.71      0.62      0.66

=== Sample Applicant Prediction ===
  Default Probability : 58.3%
  Risk Level          : Medium
```

---

## How to run

```bash
git clone https://github.com/adhimanjagota/credit-risk-predictor.git
cd credit-risk-predictor
pip install -r requirements.txt
python main.py
```

Outputs saved: `roc_curve.png`, `feature_importance.png`

---

## Project structure

```
credit-risk-predictor/
├── model.py            # Dataset generation, model training, evaluation, prediction
├── main.py             # Demo script
├── requirements.txt    # Dependencies
└── README.md
```

---

## Key ML concepts demonstrated

- **Random Forest & Gradient Boosting** — ensemble methods for tabular classification
- **ROC-AUC scoring** — appropriate metric for imbalanced binary classification
- **5-fold cross-validation** — robust model selection
- **sklearn Pipelines** — clean, production-style preprocessing + model chaining
- **Feature importance** — interpretability of tree-based models
- **Class balancing** — handling imbalanced default/non-default labels

---

## Tech stack

- Python 3.8+
- scikit-learn · pandas · numpy · matplotlib

---

## Author

Adhiman Jagota — Data Science & Applied Math @ University of Washington Seattle  
adhimanj@uw.edu
