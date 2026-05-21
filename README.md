# Credit Risk Predictor
 
A machine learning pipeline that predicts the probability of loan default using real-world financial features. Trains and compares three classifiers, selects the best by ROC-AUC, and generates visual evaluation reports.
 
---
 
## What it does
 
- Generates a realistic synthetic dataset of 2,500 loan applicants with 10 financial features
- Trains and compares **Random Forest**, **Gradient Boosting**, and **Logistic Regression**
- Selects best model via 5-fold cross-validation (ROC-AUC)
- Outputs a full classification report, confusion matrix, and ROC curve
- Predicts default probability for individual applicants
---
 
## Demo output
 
```
Generating loan dataset (2,500 applicants)...
 
Default rate: 1.1%
 
Training models...
 
=== Credit Risk Predictor — Model Comparison ===
 
  Random Forest             CV AUC: 0.6878   Test AUC: 0.8289
  Gradient Boosting         CV AUC: 0.6822   Test AUC: 0.7723
  Logistic Regression       CV AUC: 0.7446   Test AUC: 0.8779  <-- best
 
=== Best Model: Logistic Regression ===
 
              precision    recall  f1-score   support
 
  No Default       1.00      0.74      0.85       494
     Default       0.04      0.83      0.07         6
 
    accuracy                           0.74       500
   macro avg       0.52      0.79      0.46       500
weighted avg       0.99      0.74      0.84       500
 
Confusion Matrix:
  True Negatives : 365   False Positives: 129
  False Negatives: 1     True Positives : 5
 
=== Sample Applicant Prediction ===
  Default Probability : 60.5%
  Risk Level          : High
```
 
---
 
## How to run
 
```bash
git clone https://github.com/adhimanjagota/credit-risk-predictor.git
cd credit-risk-predictor
pip install -r requirements.txt
python main.py
```
 
Outputs saved: `roc_curve.png`
 
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
- **Class balancing** — handling imbalanced default/non-default labels
---
 
## Tech stack
 
- Python 3.8+
- scikit-learn · pandas · numpy · matplotlib
---
 
## Author
 
Adhiman Jagota — Data Science & Applied Math @ University of Washington Seattle
adhimanj@uw.edu
