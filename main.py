"""
Credit Risk Predictor — Demo
Trains models, prints evaluation report, saves plots, and runs a sample prediction.
"""

from model import generate_dataset, CreditRiskPredictor

def main():
    print("Generating loan dataset (2,500 applicants)...\n")
    df = generate_dataset(n_samples=2500)
    print(df.describe().round(2).to_string())
    print(f"\nDefault rate: {df['defaulted'].mean():.1%}\n")

    predictor = CreditRiskPredictor()
    print("Training models...\n")
    predictor.train(df)
    predictor.report()
    predictor.feature_importance_plot()
    predictor.roc_curve_plot()

    # Sample prediction
    applicant = {
        "age": 34,
        "income": 48000,
        "loan_amount": 15000,
        "loan_term_months": 36,
        "credit_score": 610,
        "debt_to_income": 0.42,
        "employment_years": 3.5,
        "num_credit_lines": 4,
        "missed_payments": 2,
        "has_mortgage": 0,
    }
    result = predictor.predict_single(applicant)
    print(f"\n=== Sample Applicant Prediction ===")
    print(f"  Default Probability : {result['default_probability']:.1%}")
    print(f"  Risk Level          : {result['risk_level']}")

if __name__ == "__main__":
    main()
