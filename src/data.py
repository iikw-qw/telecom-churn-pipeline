import numpy as np
import pandas as pd

NUMERIC = ["tenure_months", "monthly_charge", "total_charge", "support_calls", "data_gb", "late_payments"]
CATEGORICAL = ["contract", "internet", "payment_method", "paperless"]


def generate_customers(n=12000, seed=61):
    rng = np.random.default_rng(seed)
    contract = rng.choice(["month-to-month", "one-year", "two-year"], n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["fiber", "dsl", "none"], n, p=[0.45, 0.35, 0.20])
    payment = rng.choice(["card", "bank_transfer", "cash", "e-check"], n, p=[0.3, 0.3, 0.1, 0.3])
    paperless = rng.choice(["yes", "no"], n, p=[0.6, 0.4])
    tenure = np.clip(rng.gamma(2.0, 14.0, n), 1, 72).round()
    base_charge = np.where(internet == "fiber", 85, np.where(internet == "dsl", 55, 25))
    monthly = (base_charge * rng.uniform(0.8, 1.2, n)).round(2)
    total = (monthly * tenure * rng.uniform(0.95, 1.05, n)).round(2)
    calls = rng.poisson(np.where(internet == "fiber", 1.8, 1.0), n)
    data_gb = np.where(internet == "none", 0, rng.gamma(2.0, 20.0, n)).round(1)
    late = rng.poisson(0.3 + 0.2 * (payment == "e-check"), n)

    logit = (
        -1.0
        + np.where(contract == "month-to-month", 1.1, np.where(contract == "one-year", -0.2, -1.2))
        + 0.35 * (internet == "fiber")
        + 0.012 * (monthly - 55)
        - 0.035 * tenure
        + 0.28 * calls
        + 0.3 * late
        + 0.3 * (payment == "e-check")
        + 0.1 * (paperless == "yes")
    )
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charge": monthly,
            "total_charge": total,
            "support_calls": calls,
            "data_gb": data_gb,
            "late_payments": late,
            "contract": contract,
            "internet": internet,
            "payment_method": payment,
            "paperless": paperless,
            "churn": churn.astype(int),
        }
    )
