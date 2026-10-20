import runpy
import shutil

import pandas as pd
import pytest

from src import config
from src.black_scholes import prepare_market_data, simulate_bs_paths
from src.deep_hedging import load_model
from src.evaluation import rho_lambda
from src.hedging_runs import run_with_costs


def test_run_with_costs_on_500_paths():
    paths = simulate_bs_paths(
        S0=config.S0, mu=config.MU, sigma=config.SIGMA, T=config.T_TOTAL, n_steps=config.N_STEPS,
        n_paths=500, seed=config.SEED_TEST_PATHS,
    )
    market = prepare_market_data(
        paths, config.K_TARGET, config.K_HEDGE, config.T_TOTAL, config.SIGMA, config.R,
        hedge_maturity_factor=config.HEDGE_MATURITY_FACTOR,
    )
    model = load_model(config.MODELS_DIR / "hedgenet_kappa_sweep_1e-02.pt")
    run = run_with_costs(paths, market, model, kappa=0.01)
    assert run["cross_check_max_abs"] < 1e-4

    # reference values: the baseline on 500 test paths (seed 999) computed with the independent research implementation
    assert rho_lambda(run["pnl_nn"], config.RISK_AVERSION) == pytest.approx(0.22862942708625678, rel=1e-9)
    assert rho_lambda(run["pnl_rothert"], config.RISK_AVERSION) == pytest.approx(2.0671041140125253, rel=1e-9)
    assert rho_lambda(run["pnl_gn"], config.RISK_AVERSION) == pytest.approx(3.0479263058458583, rel=1e-9)
    assert int(run["traded_rothert"].sum()) == 8211
    assert int(run["region_undefined"].sum()) == 0


def test_tables_script_on_300_paths(tmp_path, monkeypatch):
    # scripts/tables.py reads models/ and writes tables/ relative to the working directory
    shutil.copytree(config.MODELS_DIR, tmp_path / "models")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "N_TEST", 300)  # smoke test only, not the thesis numbers
    runpy.run_path(str(config.ROOT / "scripts" / "tables.py"))

    out = tmp_path / "tables"
    names = [
        "table1_no_costs", "table2_main", "table3_kappa_sweep", "table4_sensitivity", "call_correlation_late_otm",
    ]
    expected = sorted([f"{n}.{ext}" for n in names for ext in ("csv", "tex")] + ["rothert_safeguards.csv"])
    assert sorted(p.name for p in out.iterdir()) == expected
    assert len(pd.read_csv(out / "table3_kappa_sweep.csv")) == 15
    assert len(pd.read_csv(out / "table4_sensitivity.csv")) == 9
    assert len(pd.read_csv(out / "rothert_safeguards.csv")) == 15
    assert len(pd.read_csv(out / "call_correlation_late_otm.csv")) == 3
    for name in names:
        assert "\\toprule" in (out / f"{name}.tex").read_text(encoding="utf-8")
