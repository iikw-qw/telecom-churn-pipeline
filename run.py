import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from src import core, data

OUT = Path("reports")


def main():
    OUT.mkdir(exist_ok=True)
    df = data.generate_customers()
    x_tr, x_te, y_tr, y_te = core.split(df)

    cv = core.cross_validate_all(x_tr, y_tr)
    best_name = cv.index[0]
    pipe = core.candidates()[best_name].fit(x_tr, y_tr)
    p = pipe.predict_proba(x_te)[:, 1]
    test_auc = float(roc_auc_score(y_te, p))
    imp = core.feature_importance(pipe, x_te, y_te)
    share, profit = core.campaign_profit_curve(y_te, p)
    best_share, best_profit = core.best_campaign_size(y_te, p)

    core.save_model(pipe, OUT / "churn_model.joblib")
    reload_check = float(abs(core.predict_risk(OUT / "churn_model.joblib", x_te.head(50)) - p[:50]).max())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(share, profit / 1000, color="black")
    axes[0].axvline(best_share, linestyle="--", color="grey")
    axes[0].set_xlabel("share of customers contacted (highest risk first)")
    axes[0].set_ylabel("expected profit, thousands")
    axes[0].set_title("Retention campaign profit")
    imp.head(8)[::-1].plot.barh(ax=axes[1], color="grey")
    axes[1].set_title("Permutation importance (AUC drop)")
    fig.tight_layout()
    fig.savefig(OUT / "churn.png", dpi=130)

    cv.round(4).to_csv(OUT / "cv_results.csv")
    summary = {
        "customers": int(len(df)),
        "churn_rate": round(float(df["churn"].mean()), 3),
        "cv_auc": {k: round(float(v), 4) for k, v in cv["auc_mean"].items()},
        "best_model": best_name,
        "test_auc": round(test_auc, 4),
        "top_features": list(imp.index[:4]),
        "best_campaign_share": round(best_share, 3),
        "expected_profit_at_best_share": round(best_profit, 0),
        "saved_model_reload_max_diff": reload_check,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
