"""
generate_dataset.py
-------------------
Generates a realistic synthetic bank loan dataset and saves it as CSV.
Run this once before any analysis or model training.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 1000

def generate_loan_data(n: int = N) -> pd.DataFrame:
    # Demographics
    gender        = np.random.choice(["Male", "Female"], size=n, p=[0.65, 0.35])
    married       = np.random.choice(["Yes", "No"],      size=n, p=[0.65, 0.35])
    dependents    = np.random.choice(["0", "1", "2", "3+"], size=n, p=[0.40, 0.22, 0.22, 0.16])
    education     = np.random.choice(["Graduate", "Not Graduate"], size=n, p=[0.78, 0.22])
    self_employed = np.random.choice(["Yes", "No"], size=n, p=[0.14, 0.86])

    # Financials
    applicant_income   = np.random.lognormal(mean=8.5, sigma=0.6, size=n).astype(int)
    coapplicant_income = np.where(married == "Yes",
                                  np.random.lognormal(mean=7.2, sigma=0.9, size=n).astype(int),
                                  0)
    loan_amount        = (applicant_income * np.random.uniform(0.8, 3.0, size=n) / 1000).astype(int)
    loan_amount_term   = np.random.choice([120, 180, 240, 300, 360, 480], size=n,
                                          p=[0.04, 0.08, 0.05, 0.10, 0.68, 0.05])
    credit_history     = np.random.choice([1.0, 0.0], size=n, p=[0.84, 0.16])
    property_area      = np.random.choice(["Urban", "Semiurban", "Rural"], size=n,
                                          p=[0.38, 0.38, 0.24])

    # Introduce ~5% missing values naturally
    def add_missing(arr, frac=0.05):
        arr = arr.astype(object)
        idx = np.random.choice(len(arr), size=int(frac * len(arr)), replace=False)
        arr[idx] = np.nan
        return arr

    loan_amount      = add_missing(loan_amount.astype(object))
    credit_history   = add_missing(credit_history.astype(object))
    self_employed    = add_missing(np.array(self_employed))
    dependents       = add_missing(np.array(dependents))

    # Target: loan approved (1) or rejected (0)
    # Credit history has the highest influence
    base_prob = (
        0.10
        + 0.50 * (credit_history == 1).astype(float)
        + 0.10 * (education == "Graduate").astype(float)
        + 0.08 * (married == "Yes").astype(float)
        + 0.06 * (property_area == "Semiurban").astype(float)
        + 0.04 * (property_area == "Urban").astype(float)
    )
    # Handle NaN in base_prob
    base_prob = np.where(np.isnan(base_prob.astype(float)), 0.5, base_prob.astype(float))
    base_prob = np.clip(base_prob, 0.05, 0.95)
    loan_status = np.where(np.random.rand(n) < base_prob, "Y", "N")

    loan_id = [f"LP{str(i).zfill(6)}" for i in range(1, n + 1)]

    df = pd.DataFrame({
        "Loan_ID":           loan_id,
        "Gender":            gender,
        "Married":           married,
        "Dependents":        dependents,
        "Education":         education,
        "Self_Employed":     self_employed,
        "ApplicantIncome":   applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount":        loan_amount,
        "Loan_Amount_Term":  loan_amount_term,
        "Credit_History":    credit_history,
        "Property_Area":     property_area,
        "Loan_Status":       loan_status,
    })
    return df


if __name__ == "__main__":
    import os, pathlib
    out_dir = pathlib.Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = generate_loan_data()
    out_path = out_dir / "loan_data.csv"
    df.to_csv(out_path, index=False)
    print(f"✅  Dataset saved  →  {out_path}")
    print(f"    Shape          :  {df.shape}")
    print(f"    Approval rate  :  {(df['Loan_Status'] == 'Y').mean():.1%}")
