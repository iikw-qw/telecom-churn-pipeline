import numpy as np
import pytest

from src import core, data


@pytest.fixture(scope="module")
def fitted():
    df = data.generate_customers(n=5000, seed=1)
    x_tr, x_te, y_tr, y_te = core.split(df)
    pipe = core.candidates()["logistic"].fit(x_tr, y_tr)
    return pipe, x_te, y_te


def test_pipeline_handles_unseen_category(fitted):
    pipe, x_te, _ = fitted
    row = x_te.head(1).copy()
    row["payment_method"] = "crypto"
    assert 0 <= pipe.predict_proba(row)[0, 1] <= 1


def test_model_beats_chance(fitted):
    from sklearn.metrics import roc_auc_score

    pipe, x_te, y_te = fitted
    assert roc_auc_score(y_te, pipe.predict_proba(x_te)[:, 1]) > 0.7


def test_profit_curve_known_case():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.3, 0.1])
    share, profit = core.campaign_profit_curve(y, p, retained_value=100, offer_cost=10, save_rate=0.5)
    assert share[-1] == 1.0
    assert profit[1] == pytest.approx(2 * 0.5 * 100 - 2 * 10)
    assert core.best_campaign_size(y, p, retained_value=100, offer_cost=10, save_rate=0.5)[0] == pytest.approx(0.5)


def test_saved_model_gives_same_predictions(fitted, tmp_path):
    pipe, x_te, _ = fitted
    core.save_model(pipe, tmp_path / "m.joblib")
    again = core.predict_risk(tmp_path / "m.joblib", x_te.head(20))
    assert np.allclose(again, pipe.predict_proba(x_te.head(20))[:, 1])


def test_contract_type_matters(fitted):
    pipe, x_te, y_te = fitted
    imp = core.feature_importance(pipe, x_te, y_te, repeats=3)
    assert "contract" in list(imp.index[:3])
