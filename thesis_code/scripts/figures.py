"""Draws the eleven thesis figures into figures/ as PDF and PNG.

Run from the thesis_code folder: python -m scripts.figures (py -m scripts.figures on Windows), takes about 20 s. The networks are
loaded from MODELS_DIR, nothing is trained. Each figure has its own section below; comment a section
out to skip that figure. The test paths, the baseline run and the kappa sweep at the top are shared
by several figures.
"""

import numpy as np

from src.benchmarks import single_path_positions
from src.black_scholes import prepare_market_data, simulate_bs_paths
from src.config import (
    HEDGE_MATURITY_FACTOR, K_HEDGE, K_TARGET, MU, N_STEPS, N_STEPS_FINE, N_TEST, R, RISK_AVERSION, S0,
    SEED_TEST_PATHS, SIGMA, T_TOTAL,
)
from src.deep_hedging import load_model
from src.evaluation import gamma_move_map, hourglass_panels, rho_lambda
from src.hedging_runs import (
    decision_surface, delta_curve, print_run_summary, run_single_asset_no_costs,
    run_single_asset_with_costs, run_two_assets_no_costs, run_with_costs,
)
from src.plotting import (
    plot_delta, plot_heatmap, plot_hourglass, plot_kappa_sweep, plot_mean_positions, plot_pnl_with_costs,
    plot_position_space, plot_robustness, plot_stock_hedge_costs, save_figure,
)

MODELS_DIR = "models"  # use "models_retrained" to draw the figures with retrained networks

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


# --- Figure 1: NN delta against Black-Scholes delta, one asset, no costs ---
model = load_model(f"{MODELS_DIR}/hedgenet_nb01_entropic_l1.pt")
pnl_nn, pnl_delta, pnl_unhedged = run_single_asset_no_costs(paths, model)
print(f"P&L std: NN {pnl_nn.std():.4f}, BS delta {pnl_delta.std():.4f}, unhedged {pnl_unhedged.std():.4f}")
s_grid = np.linspace(85, 115, 200)
delta_nn, delta_bs = delta_curve(model, s_grid, k=15, n_steps=N_STEPS)  # step 15 = half way to maturity
print(f"max |NN delta - BS delta| at step 15: {np.abs(delta_nn - delta_bs).max():.6f}")
fig = plot_delta(s_grid, delta_nn, delta_bs)
save_figure(fig, "delta_nn_vs_bs")


# --- Figure 2: positions of NN and Gamma-neutral hedge, no costs ---
model = load_model(f"{MODELS_DIR}/hedgenet_nb07_mse_3feat.pt")
pnl_nn, stock_nn, call_nn, pnl_gn, stock_gn, call_gn = run_two_assets_no_costs(paths, market, model)
print(f"P&L std: NN {pnl_nn.std():.4f}, Gamma-neutral {pnl_gn.std():.4f}")
positions = single_path_positions(
    path_idx=1, steps=[0, 5, 10, 15, 20, 25], nn_stock=stock_nn, nn_call=call_nn,
    gn_stock=stock_gn, gn_call=call_gn, delta_target=market.delta_target, delta_hedge=market.delta_hedge,
)
print(positions.round(4))
fig = plot_position_space(positions)
save_figure(fig, "position_space_no_costs")


# --- Figure 3: mean positions of NN and Gamma-neutral hedge over time, no costs ---
model = load_model(f"{MODELS_DIR}/hedgenet_nb07_mse_3feat.pt")
pnl_nn, stock_nn, call_nn, pnl_gn, stock_gn, call_gn = run_two_assets_no_costs(paths, market, model)
gap_stock = np.abs(stock_nn.mean(axis=0) - stock_gn.mean(axis=0))
gap_call = np.abs(call_nn.mean(axis=0) - call_gn.mean(axis=0))
print(f"max |mean NN - mean Gamma-neutral| over k: stock {gap_stock.max():.4f}, call {gap_call.max():.4f}")
fig = plot_mean_positions(stock_nn, call_nn, stock_gn, call_gn, band=(0.1, 0.9))
save_figure(fig, "mean_positions_no_costs")


# --- Figure 4: one-asset network with costs: P&L, costs and decision surface ---
model = load_model(f"{MODELS_DIR}/hedgenet_nb05_entropic_kappa1e-02.pt")
pnl, costs = run_single_asset_with_costs(paths, model, kappa=0.01)
print(f"mean P&L {pnl.mean():.4f}, mean costs {costs.mean():.4f}")
ss, dd, zz = decision_surface(
    model, s_grid=np.linspace(85, 115, 60), prev_grid=np.linspace(0.0, 1.0, 60), k=15, n_steps=N_STEPS
)
fig = plot_stock_hedge_costs(pnl, costs, ss, dd, zz, step=15)
save_figure(fig, "stock_hedge_costs")


# --- Figure 5: no-trade region at four trades of the median path, baseline ---
panels, path_idx, n_trades, steps = hourglass_panels(
    base["region"], base["traded_rothert"], base["stock_rothert"], base["call_rothert"],
    base["stock_nn"], base["call_nn"], epsilon=0.01, n_panels=4, n_points=401,
)
print(f"median path {path_idx} with {n_trades} Rothert trades, panel steps {steps}")
fig = plot_hourglass(panels)
save_figure(fig, "hourglass_region")


# --- Figure 6: terminal P&L of NN, Rothert and Gamma-neutral hedge with costs ---
window = (-5.0, 2.0)  # drawn P&L range, paths outside are counted, not drawn
print_run_summary("baseline", base, window)
fig = plot_pnl_with_costs(base["pnl_nn"], base["pnl_rothert"], base["pnl_gn"], window)
save_figure(fig, "pnl_with_costs")


# --- Figure 7: positions of NN and Gamma-neutral hedge with costs ---
positions = single_path_positions(
    path_idx=4, steps=[0, 5, 10, 15, 20, 25], nn_stock=base["stock_nn"], nn_call=base["call_nn"],
    gn_stock=base["stock_gn"], gn_call=base["call_gn"], delta_target=market.delta_target,
    delta_hedge=market.delta_hedge,
)
print(positions.round(4))
fig = plot_position_space(positions)
save_figure(fig, "position_space_with_costs")


# --- Figure 8: mean normalised parallel adjustment |x_after - x_before|, NN and Rothert ---
heatmap_kappas = [1e-2, 1e-3, 1e-4]
maps_nn, maps_rothert = [], []
for kappa in heatmap_kappas:
    run = sweep[kappa]
    map_nn, edges = gamma_move_map(
        run["region"], paths, run["stock_nn"], run["call_nn"],
        epsilon=kappa, window=3.0, n_bins=60, min_count=20,
    )
    map_rothert, edges = gamma_move_map(
        run["region"], paths, run["stock_rothert"], run["call_rothert"],
        epsilon=kappa, window=3.0, n_bins=60, min_count=20,
        check_no_trade_inside=True,  # Rothert never trades from inside its region, checked here
    )
    maps_nn.append(map_nn)
    maps_rothert.append(map_rothert)
all_cells = np.concatenate([m[np.isfinite(m)] for m in maps_nn + maps_rothert])  # cells with too few observations are NaN
color_max = float(np.percentile(all_cells, 95.0))  # colour scale ends at the 95th percentile, the arrow on the colour bar marks the clip
print(f"hourglass and no-trade cross-checks passed; colour scale maximum {color_max:.6f}")
fig = plot_heatmap(heatmap_kappas, maps_nn, maps_rothert, edges, color_max)
save_figure(fig, "heatmap_gamma_move")


# --- Figure 9: rho over kappa and the ratio NN / Rothert ---
kappas = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
rho_nn, rho_rothert, rho_gn = [], [], []
for kappa in kappas:
    run = sweep[kappa]
    rho_nn.append(rho_lambda(run["pnl_nn"], RISK_AVERSION))
    rho_rothert.append(rho_lambda(run["pnl_rothert"], RISK_AVERSION))
    rho_gn.append(rho_lambda(run["pnl_gn"], RISK_AVERSION))
    print_run_summary(f"kappa {kappa:g}", run)
fig = plot_kappa_sweep(kappas, rho_nn, rho_rothert, rho_gn)
save_figure(fig, "kappa_sweep")


# --- Figure 10: robustness, 30 against 120 trading steps ---
paths_fine = simulate_bs_paths(
    S0=S0, mu=MU, sigma=SIGMA, T=T_TOTAL, n_steps=N_STEPS_FINE, n_paths=N_TEST, seed=SEED_TEST_PATHS
)
market_fine = prepare_market_data(
    paths_fine, K_TARGET, K_HEDGE, T_TOTAL, SIGMA, R, hedge_maturity_factor=HEDGE_MATURITY_FACTOR
)
model = load_model(f"{MODELS_DIR}/hedgenet_nb09_n120_kappa0.01.pt")
fine = run_with_costs(paths_fine, market_fine, model, kappa=0.01)
window = (-8.0, 2.0)
print_run_summary("baseline", base, window)
print_run_summary("fine grid", fine, window)
fig = plot_robustness(
    base["pnl_nn"], base["pnl_rothert"], base["pnl_gn"],  # left panel
    fine["pnl_nn"], fine["pnl_rothert"], fine["pnl_gn"],  # right panel
    titles=("(a) Baseline: 30 steps", "(b) Fine grid: 120 steps"), window=window,
)
save_figure(fig, "robustness_time_grid")


# --- Figure 11: robustness, cost basis of the call leg ---
model = load_model(f"{MODELS_DIR}/hedgenet_nb10_n30_kappa0.01_deltaneutral.pt")
delta_neutral = run_with_costs(paths, market, model, kappa=0.01, cost_model="delta_neutral")
window = (-8.0, 2.0)
print_run_summary("baseline", base, window)
print_run_summary("delta-neutral call leg", delta_neutral, window)
fig = plot_robustness(
    base["pnl_nn"], base["pnl_rothert"], base["pnl_gn"],  # left panel
    delta_neutral["pnl_nn"], delta_neutral["pnl_rothert"], delta_neutral["pnl_gn"],  # right panel
    titles=(
        r"(a) Baseline: call leg $\kappa\,C_k$",
        r"(b) Delta-neutral: call leg $\kappa\,\Delta^{\mathrm{hedge}}_k S_k$",
    ),
    window=window,
)
save_figure(fig, "robustness_cost_model")

print("done.")
