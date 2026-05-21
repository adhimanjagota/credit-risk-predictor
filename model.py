"""
Credit Risk Predictor
Predicts the likelihood of loan default using a Random Forest classifier.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def generate_dataset(n_samples: int = 2500, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic loan dataset.
    Features mirror real-world credit risk factors.
    """
    rng = np.random.RandomState(random_state)

    age               = rng.randint(21, 65, n_samples)
    income            = rng.normal(55000, 20000, n_samples).clip(15000, 200000)
    loan_amount       = rng.normal(12000, 7000, n_samples).clip(1000, 50000)
    loan_term_months  = rng.choice([12, 24, 36, 48, 60], n_samples)
    credit_score      = rng.normal(650, 80, n_samples).clip(300, 850).astype(int)
    debt_to_income    = rng.uniform(0.05, 0.65, n_samples)
    employment_years  = rng.exponential(5, n_samples).clip(0, 30)
    num_credit_lines  = rng.randint(1, 15, n_samples)
    missed_payments   = rng.choice([0, 1, 2, 3, 4, 5], n_samples,
                                    p=[0.55, 0.20, 0.12, 0.07, 0.04, 0.02])
    has_mortgage      = rng.randint(0, 2, n_samples)

    # Default probability driven by realistic risk factors
    default_score = (
        - 0.003  * credit_score
        + 0.8    * debt_to_income
        + 0.15   * missed_payments
        - 0.005  * employment_years
        - 0.000003 * income
        + 0.000008 * loan_amount
        + rng.normal(0, 0.15, n_samples)
    )
    default_prob  = 1 / (1 + np.exp(-default_score * 3))
    defaulted     = (rng.uniform(0, 1, n_samples) < default_prob).astype(int)

    return pd.DataFrame({
        "age":              age,
        "income":           income.astype(int),
        "loan_amount":      loan_amount.astype(int),
        "loan_term_months": loan_term_months,
        "credit_score":     credit_score,
        "debt_to_income":   debt_to_income.round(3),
        "employment_years": employment_years.round(1),
        "num_credit_lines": num_credit_lines,
        "missed_payments":  missed_payments,
        "has_mortgage":     has_mortgage,
        "defaulted":        defaulted,
    })


class CreditRiskPredictor:
    """
    Trains and evaluates multiple classifiers.
    Selects best model by ROC-AUC via cross-validation.
    """

    MODELS = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=10,
            class_weight="balanced", random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
    }

    FEATURES = [
        "age", "income", "loan_amount", "loan_term_months",
        "credit_score", "debt_to_income", "employment_years",
        "num_credit_lines", "missed_payments", "has_mortgage",
    ]

    def __init__(self):
        self.best_model_name = None
        self.best_pipeline   = None
        self.X_test          = None
        self.y_test          = None
        self.results         = {}

    def train(self, df: pd.DataFrame):
        X = df[self.FEATURES]
        y = df["defaulted"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        best_auc = 0
        for name, clf in self.MODELS.items():
            needs_scaling = "Logistic" in name
            steps = []
            if needs_scaling:
                steps.append(("scaler", StandardScaler()))
            steps.append(("clf", clf))
            pipeline = Pipeline(steps)

            cv_auc = cross_val_score(
                pipeline, self.X_train, self.y_train,
                cv=5, scoring="roc_auc"
            ).mean()

            pipeline.fit(self.X_train, self.y_train)
            test_auc = roc_auc_score(self.y_test, pipeline.predict_proba(self.X_test)[:, 1])
            self.results[name] = {"cv_auc": cv_auc, "test_auc": test_auc, "pipeline": pipeline}

            if test_auc > best_auc:
                best_auc = test_auc
                self.best_model_name = name
                self.best_pipeline   = pipeline

    def report(self):
        print("=== Credit Risk Predictor — Model Comparison ===\n")
        for name, res in self.results.items():
            tag = " <-- best" if name == self.best_model_name else ""
            print(f"  {name:<25} CV AUC: {res['cv_auc']:.4f}   Test AUC: {res['test_auc']:.4f}{tag}")

        print(f"\n=== Best Model: {self.best_model_name} ===\n")
        y_pred = self.best_pipeline.predict(self.X_test)
        print(classification_report(self.y_test, y_pred, target_names=["No Default", "Default"]))

        cm = confusion_matrix(self.y_test, y_pred)
        print("Confusion Matrix:")
        print(f"  True Negatives : {cm[0,0]}   False Positives: {cm[0,1]}")
        print(f"  False Negatives: {cm[1,0]}   True Positives : {cm[1,1]}")

    def feature_importance_plot(self):
        clf = self.best_pipeline.named_steps["clf"]
        if not hasattr(clf, "feature_importances_"):
            print("Feature importance not available for this model.")
            return

        importances = pd.Series(clf.feature_importances_, index=self.FEATURES)
        importances = importances.sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(importances.index, importances.values, color="#378ADD", edgecolor="none")
        ax.set_xlabel("Importance", fontsize=11)
        ax.set_title(f"Feature Importances — {self.best_model_name}", fontsize=13)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=150)
        print("\nSaved: feature_importance.png")

    def roc_curve_plot(self):
        fig, ax = plt.subplots(figsize=(6, 5))
        for name, res in self.results.items():
            proba = res["pipeline"].predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, proba)
            ax.plot(fpr, tpr, label=f"{name} (AUC={res['test_auc']:.3f})")

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — All Models")
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig("roc_curve.png", dpi=150)
        print("Saved: roc_curve.png")

    def predict_single(self, applicant: dict) -> dict:
        """Predict default risk for a single loan applicant."""
        df = pd.DataFrame([applicant])[self.FEATURES]
        prob = self.best_pipeline.predict_proba(df)[0][1]
        risk = "High" if prob > 0.6 else "Medium" if prob > 0.35 else "Low"
        return {"default_probability": round(prob, 4), "risk_level": risk}
