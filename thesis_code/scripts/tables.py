"""Writes the four thesis tables and the Rothert safeguard counts into tables/ as CSV and TeX.

Run from the thesis_code folder: python -m scripts.tables (py -m scripts.tables on Windows), takes about 17 s. The networks are
loaded from MODELS_DIR, nothing is trained. The CSV files hold all columns, the TeX files a compact
subset as a bare booktabs tabular. Each table has its own section below; comment a section out to
skip that table. The test paths and the runs with costs at the top are shared by several tables.
"""

import numpy as np
import pandas as pd

from src.black_scholes import prepare_market_data, simulate_bs_paths
from src.config import (
    HEDGE_MATURITY_FACTOR, K_HEDGE, K_TARGET, MU, N_STEPS, N_STEPS_FINE, N_TEST, R, S0,
    SEED_TEST_PATHS, SIGMA, T_TOTAL,
)
from src.deep_hedging import load_model
from src.evaluation import call_correlation_late_otm, pnl_statistics, save_table, strategy_metrics
from src.hedging_runs import run_single_asset_no_costs, run_two_assets_no_costs, run_with_costs

MODELS_DIR = "models"  # use "models_retrained" to build the tables with retrained networks
TABLES_DIR = "tables"

# 1) test paths, identical for all strategies
paths = simulate_bs_paths(
    S0=S0, mu=MU, sigma=SIGMA, T=T_TOTAL, n_steps=N_STEPS, n_paths=N_TEST, seed=SEED_TEST_PATHS
)
market = prepare_market_data(
    paths, K_TARGET, K_HEDGE, T_TOTAL, SIGMA, R, hedge_maturity_factor=HEDGE_MATURITY_FACTOR
)

# 2) baseline: kappa = 0.01, 30 steps, proportional costs on both legs
model = load_model(f"{MODELS_DIR}/hedgenet_kappa_sweep_1e-02.pt")
base = run_with_costs(paths, market, model, kappa=0.01)

# 3) kappa sweep, one network per cost rate; the baseline is the kappa = 0.01 point
sweep = {0.01: base}
for kappa in [3e-3, 1e-3, 3e-4, 1e-4]:
    model = load_model(f"{MODELS_DIR}/hedgenet_kappa_sweep_{kappa:.0e}.pt")
    sweep[kappa] = run_with_costs(paths, market, model, kappa=kappa)

# 4) robustness: 120 trading steps, and call-leg costs on delta_hedge * S instead of the call price
paths_fine = simulate_bs_paths(
    S0=S0, mu=MU, sigma=SIGMA, T=T_TOTAL, n_steps=N_STEPS_FINE, n_paths=N_TEST, seed=SEED_TEST_PATHS
)
market_fine = prepare_market_data(
    paths_fine, K_TARGET, K_HEDGE, T_TOTAL, SIGMA, R, hedge_maturity_factor=HEDGE_MATURITY_FACTOR
)
model = load_model(f"{MODELS_DIR}/hedgenet_nb09_n120_kappa0.01.pt")
fine = run_with_costs(paths_fine, market_fine, model, kappa=0.01)
model = load_model(f"{MODELS_DIR}/hedgenet_nb10_n30_kappa0.01_deltaneutral.pt")
delta_neutral = run_with_costs(paths, market, model, kappa=0.01, cost_model="delta_neutral")
scenarios = [  # label in Table 4 and in the safeguard counts, run
    ("Baseline (30 steps)", base),
    ("Fine grid (120 steps)", fine),
    ("Delta-neutral call cost basis", delta_neutral),
]


# --- Table 1: no costs, from the unhedged position to two instruments ---
model = load_model(f"{MODELS_DIR}/hedgenet_nb01_entropic_l1.pt")
pnl_nn_stock, pnl_delta, pnl_unhedged = run_single_asset_no_costs(paths, model)
model = load_model(f"{MODELS_DIR}/hedgenet_nb07_mse_3feat.pt")
pnl_nn_two, _, call_nn_two, pnl_gn, _, call_gn = run_two_assets_no_costs(paths, market, model)
rows = []
for strategy, instruments, pnl in [
    ("Unhedged", "none", pnl_unhedged),
    ("Black-Scholes delta", "stock", pnl_delta),
    ("NN (stock only)", "stock", pnl_nn_stock),
    ("Gamma-neutral", "stock + call", pnl_gn),
    ("NN (stock + call)", "stock + call", pnl_nn_two),
]:
    stat = pnl_statistics(pnl)  # mean, std, median, min, max, q01, q05, q95, mean_abs
    rows.append({"strategy": strategy, "instruments": instruments, **{f"pnl_{k}": v for k, v in stat.items()}})
table1 = pd.DataFrame(rows)
print(table1.round(4).to_string(index=False))
save_table(
    table1, "table1_no_costs", out_dir=TABLES_DIR,
    tex_columns=[
        ("Strategy", "strategy", None), ("Instruments", "instruments", None),
        ("Mean", "pnl_mean", ".4f"), ("Std", "pnl_std", ".4f"),
        (r"1\% quantile", "pnl_q01", ".4f"), (r"5\% quantile", "pnl_q05", ".4f"),
    ],
)


# --- Call correlation NN vs. Gamma-neutral without costs, with and without the late-OTM paths ---
# late-OTM: spot below 98 at any time index from step 25 on; uses the no-cost run of Table 1
table_corr = pd.DataFrame(call_correlation_late_otm(paths, call_nn_two, call_gn, step_from=25, spot_threshold=98.0))
print(table_corr.round(4).to_string(index=False))
save_table(
    table_corr, "call_correlation_late_otm", out_dir=TABLES_DIR,
    tex_columns=[
        ("Paths", "subset", None), ("Number", "n_paths", "d"), ("Share", "share_paths", ".4f"),
        ("Correlation NN/Gamma-neutral call", "corr_call", ".3f"),
    ],
)


# --- Table 2: NN, Rothert and Gamma-neutral hedge at kappa = 0.01 ---
rows = []
for strategy, block, short in [
    ("NN", "main", "nn"),
    ("Rothert", "main", "rothert"),
    ("Gamma-neutral", "context", "gn"),  # context only, the Gamma-neutral hedge does not minimise rho
]:
    rows.append({"block": block, "strategy": strategy, **strategy_metrics(base, short)})
table2 = pd.DataFrame(rows)
print(table2.round(4).to_string(index=False))
save_table(
    table2, "table2_main", out_dir=TABLES_DIR,
    tex_columns=[
        ("Strategy", "strategy", None), ("Mean P\\&L", "pnl_mean", ".4f"), ("Std", "pnl_std", ".4f"),
        (r"1\% quantile", "pnl_q01", ".4f"), (r"$\rho_a$", "rho", ".4f"),
        ("Mean cost", "cost_mean", ".4f"), ("Trades per path", "mean_n_trades", ".2f"),
        ("No-trade share", "pct_steps_no_trade", ".4f"), ("Stock turnover", "mean_stock_turnover", ".4f"),
        ("Call turnover", "mean_call_turnover", ".4f"),
    ],
)


# --- Table 3: rho over kappa ---
rows, tex_rows = [], []
for kappa in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]:
    run = sweep[kappa]
    nn, rothert, gn = strategy_metrics(run, "nn"), strategy_metrics(run, "rothert"), strategy_metrics(run, "gn")
    ratio = nn["rho"] / rothert["rho"]
    for strategy, m in [("NN", nn), ("Rothert", rothert), ("Gamma-neutral", gn)]:
        rows.append({
            "kappa": kappa, "strategy": strategy, "rho": m["rho"], "ratio_nn_to_rothert": ratio,
            "pnl_mean": m["pnl_mean"], "pnl_std": m["pnl_std"], "cost_mean": m["cost_mean"],
            "mean_stock_turnover": m["mean_stock_turnover"], "mean_call_turnover": m["mean_call_turnover"],
            "pct_steps_no_trade": m["pct_steps_no_trade"],
        })
    tex_rows.append({  # TeX: one row per kappa
        "kappa": kappa, "rho_nn": nn["rho"], "rho_rothert": rothert["rho"], "rho_gn": gn["rho"],
        "ratio": ratio, "nn_stock_turnover": nn["mean_stock_turnover"],
        "rothert_no_trade": rothert["pct_steps_no_trade"],
    })
table3 = pd.DataFrame(rows)
print(table3.round(4).to_string(index=False))
save_table(
    table3, "table3_kappa_sweep", out_dir=TABLES_DIR, tex_frame=pd.DataFrame(tex_rows),
    tex_columns=[
        (r"$\kappa$", "kappa", "g"), (r"$\rho_a$ NN", "rho_nn", ".4f"),
        (r"$\rho_a$ Rothert", "rho_rothert", ".4f"), (r"$\rho_a$ Gamma-neutral", "rho_gn", ".4f"),
        ("NN/Rothert", "ratio", ".4f"), ("NN stock turnover", "nn_stock_turnover", ".4f"),
        ("Rothert no-trade share", "rothert_no_trade", ".4f"),
    ],
)


# --- Table 4: sensitivity to the time grid and to the call-leg cost basis ---
rows = []
for label, run in scenarios:
    nn, rothert, gn = strategy_metrics(run, "nn"), strategy_metrics(run, "rothert"), strategy_metrics(run, "gn")
    ratio = nn["rho"] / rothert["rho"]
    for strategy, m in [("NN", nn), ("Rothert", rothert), ("Gamma-neutral", gn)]:
        rows.append({
            "scenario": label, "strategy": strategy, "pnl_mean": m["pnl_mean"], "pnl_std": m["pnl_std"],
            "pnl_min": m["pnl_min"], "pnl_max": m["pnl_max"], "rho": m["rho"], "cost_mean": m["cost_mean"],
            "mean_stock_turnover": m["mean_stock_turnover"], "mean_call_turnover": m["mean_call_turnover"],
            "ratio_nn_to_rothert": ratio,
        })
table4 = pd.DataFrame(rows)
print(table4.round(4).to_string(index=False))
table4_tex = table4.copy()
table4_tex.loc[table4["strategy"] != "NN", "scenario"] = ""  # label only in the first row of a scenario
table4_tex.loc[table4["strategy"] != "NN", "ratio_nn_to_rothert"] = np.nan  # ratio only in the NN row
save_table(
    table4, "table4_sensitivity", out_dir=TABLES_DIR, tex_frame=table4_tex,
    tex_columns=[
        ("Scenario", "scenario", None), ("Strategy", "strategy", None), ("Mean", "pnl_mean", ".4f"),
        ("Std", "pnl_std", ".4f"), (r"$\rho_a$", "rho", ".4f"), ("Mean cost", "cost_mean", ".4f"),
        ("Stock turnover", "mean_stock_turnover", ".4f"), ("Call turnover", "mean_call_turnover", ".4f"),
        ("NN/Rothert", "ratio_nn_to_rothert", ".4f"),
    ],
)


# --- Rothert safeguards: how often they and the undefined-boundary rule were active ---
rows = []
for label, run in scenarios:
    for quantity, grid, flags in [
        ("gamma_problematic", "time grid", run["region"].gamma_problematic),  # diagnostic only
        ("gamma_floor_active", "time grid", run["region"].gamma_floor_active),
        ("stock_cap_active", "trading grid", run["stock_cap_active"]),
        ("call_cap_active", "trading grid", run["call_cap_active"]),
        ("region_undefined", "trading grid", run["region_undefined"]),
    ]:
        n_active = int(flags.sum())
        rows.append({
            "scenario": label, "quantity": quantity, "grid": grid, "n_active": n_active,
            "n_observations": int(flags.size), "share": n_active / flags.size,
        })
safeguards = pd.DataFrame(rows)
print(safeguards.to_string(index=False))
save_table(safeguards, "rothert_safeguards", out_dir=TABLES_DIR)  # CSV only

print("done.")
