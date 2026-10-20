import numpy as np
import pytest

from src.evaluation import call_correlation_late_otm, rho_lambda


def test_rho_lambda():
    assert rho_lambda(np.full(100, 0.7), 2.0) == pytest.approx(-0.7, rel=1e-12)

    x = np.random.default_rng(1).normal(size=2000)
    expected = np.log(np.mean(np.exp(-2.0 * x))) / 2.0
    assert rho_lambda(x, 2.0) == pytest.approx(expected, rel=1e-10)

    # a smaller P&L is a larger loss
    assert rho_lambda(x - 1.0, 2.0) > rho_lambda(x, 2.0)


def test_call_correlation_late_otm():
    # path 0 drops below 98 at time index 2, paths 1-2 stay at or above 98 from index 1 on
    paths = np.array([[100.0, 99.0, 97.0], [100.0, 98.0, 105.0], [100.0, 101.0, 102.0]])
    call_nn = np.array([[0.0, 1.0], [1.0, 2.0], [3.0, 4.0]])
    call_gn = np.array([[5.0, -5.0], [2.0, 4.0], [6.0, 8.0]])
    rows = call_correlation_late_otm(paths, call_nn, call_gn, step_from=1, spot_threshold=98.0)

    assert [r["subset"] for r in rows] == ["All paths", "Without late-OTM paths", "Late-OTM paths only"]
    assert [r["n_paths"] for r in rows] == [3, 2, 1]
    assert rows[1]["share_paths"] == pytest.approx(2 / 3)
    assert rows[1]["corr_call"] == pytest.approx(1.0)  # call_gn = 2 * call_nn on the kept paths
    assert rows[0]["corr_call"] == pytest.approx(np.corrcoef(call_nn.ravel(), call_gn.ravel())[0, 1])
    assert rows[2]["corr_call"] == pytest.approx(-1.0)
