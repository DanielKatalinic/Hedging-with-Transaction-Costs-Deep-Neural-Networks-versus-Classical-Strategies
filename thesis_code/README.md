# Rothert hourglass benchmark: code for the thesis figures and tables

This repository reproduces the eleven figures and four tables of the thesis comparison between a deep
hedging network and the formula-based no-trade region of N. Rothert. A sold call option is hedged with
the stock and a tradable hedge call under proportional transaction costs. The network is compared with
Rothert's hourglass strategy and with a Gamma-neutral hedge on identical test paths, prices, initial
premium, cost rates, liquidation rule and P&L function. The code only loads the shipped networks by
default; retraining them is a separate, explicit step (`scripts/train.py`).

## Setup and commands

```
git clone <repository-url>
cd <repository>
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # Linux / macOS
pip install -r requirements.txt
```

On Windows the launcher `py` can be used instead of `python`. All commands run from the repository
root (the directory that contains `src/` and `scripts/`). The three scripts are started as modules
(`-m scripts.figures`), so Python finds `src/` without any path setup.

```
python -m scripts.figures                               # all 11 figures into figures/ (pdf + png), about 20 s

python -m scripts.tables                                # 4 tables and the call correlation (csv + tex), rothert_safeguards.csv into tables/, about 17 s

python -m scripts.train                                 # retrains all 10 networks into models_retrained/, about 55 min

python -m pytest -q                                     # 13 tests
```

`scripts/figures.py` and `scripts/tables.py` never train. Both load the checkpoints named in their sections from
`MODELS_DIR` (set at the top of each file) with `load_model`, which does not check the stored
fingerprint; set `MODELS_DIR = "models_retrained"` to use retrained networks (a shortened retrain then
gives figures and tables for inspection only, not the thesis numbers). To skip a figure or a table,
comment out its section. `scripts/train.py` writes to `OUT_DIR = "models_retrained"` and stops before training
if `OUT_DIR` points to `models/`, so the shipped checkpoints are never overwritten; a new run replaces
the files in `models_retrained/`. Remove a name from `TRAIN` at the top of `scripts/train.py` to skip that
network, and set `N_EPOCHS_OVERRIDE = 50` for a quick test.

## Runtimes

Measured on one development machine (Windows 11, Python 3.12.3, CPU only, 8 torch threads).

| command | wall-clock time |
|---|---|
| `scripts/figures.py` (all 11 figures) | about 20 s |
| `scripts/tables.py` (4 tables and the safeguard counts) | about 17 s |
| `python -m pytest -q` (13 tests) | about 9 s |
| `scripts/train.py` with `N_EPOCHS_OVERRIDE = 50` (all 10 networks) | about 35 s |
| `scripts/train.py` single_asset_costs (5000 epochs) | about 2 min |
| `scripts/train.py` single_asset_no_costs, two_assets_no_costs (15000 epochs each) | about 4.5 min each |
| `scripts/train.py` each of the five kappa_sweep networks and delta_neutral_basis (10000 epochs) | about 4.5 min each |
| `scripts/train.py` fine_grid_120 (10000 epochs, 120 steps) | about 18 min |
| `scripts/train.py` all 10 networks | about 55 min |

## Figures

`scripts/figures.py` is one linear script: the test paths, the baseline run and the kappa sweep come first,
then one section per figure in thesis order. Every run and every plot call is visible there with its
arguments. The numerical work is in `src/hedging_runs.py` and `src/evaluation.py`, the drawing in
`src/plotting.py`.

| thesis figure | file | runs (`src/hedging_runs.py`) | figure data | drawing (`src/plotting.py`) |
|---|---|---|---|---|
| 1 | `delta_nn_vs_bs` | `run_single_asset_no_costs` | `delta_curve` | `plot_delta` |
| 2 | `position_space_no_costs` | `run_two_assets_no_costs` | `benchmarks.single_path_positions` | `plot_position_space` |
| 3 | `mean_positions_no_costs` | `run_two_assets_no_costs` | | `plot_mean_positions` |
| 4 | `stock_hedge_costs` | `run_single_asset_with_costs` | `decision_surface` | `plot_stock_hedge_costs` |
| 5 | `hourglass_region` | `run_with_costs` baseline | `evaluation.hourglass_panels` | `plot_hourglass` |
| 6 | `pnl_with_costs` | `run_with_costs` baseline | | `plot_pnl_with_costs` |
| 7 | `position_space_with_costs` | `run_with_costs` baseline | `benchmarks.single_path_positions` | `plot_position_space` |
| 8 | `heatmap_gamma_move` | `run_with_costs` kappa 1e-2, 1e-3, 1e-4 | `evaluation.gamma_move_map` | `plot_heatmap` |
| 9 | `kappa_sweep` | `run_with_costs` five kappas | `evaluation.rho_lambda` | `plot_kappa_sweep` |
| 10 | `robustness_time_grid` | `run_with_costs` baseline and 120 steps | | `plot_robustness` |
| 11 | `robustness_cost_model` | `run_with_costs` baseline and delta-neutral call leg | | `plot_robustness` |

For every run with costs `print_run_summary` prints P&L mean and standard deviation, rho and mean cost
per strategy, rho NN / rho Rothert, the number of paths outside the drawn P&L window (they are not
drawn), the number of Rothert observations with an undefined region boundary, how often the gamma
floor and the two position caps were active (never: they are off) and the largest deviation of the
Rothert cross-check (the Rothert positions are re-priced through the network's P&L chain and must agree
within `rtol = atol = 1e-5`).

## Tables

`scripts/tables.py` is one linear script like `scripts/figures.py`: the test paths and the runs with costs come first,
then one section per table. Each section builds its DataFrame with the columns visible, prints it and
writes it with `evaluation.save_table` (CSV always, TeX where columns are given).

All tables use 25 000 test paths (seed 999). CSV files hold the full metric set, TeX files a compact
subset as a bare `booktabs` tabular without float or caption, so they can be pulled in with `\input`
(add `\usepackage{booktabs}`). Numeric cells are set in math mode so that minus signs typeset.

| table | files | runs (`src/hedging_runs.py`) | metrics (`src/evaluation.py`) | content |
|---|---|---|---|---|
| 1 | `table1_no_costs` | `run_single_asset_no_costs` and `run_two_assets_no_costs` | `pnl_statistics` | unhedged, Black-Scholes delta, network (stock only), Gamma-neutral, network (stock + call); no costs |
| 2 | `table2_main` | `run_with_costs` baseline | `strategy_metrics` | NN, Rothert, Gamma-neutral at kappa = 0.01: P&L, cost and trading statistics, rho |
| 3 | `table3_kappa_sweep` | `run_with_costs` five kappas | `strategy_metrics` | rho over kappa in {1e-2, 3e-3, 1e-3, 3e-4, 1e-4}; TeX one row per kappa |
| 4 | `table4_sensitivity` | `run_with_costs` baseline, 120 steps and delta-neutral call leg | `strategy_metrics` | baseline, fine time grid (120 steps), delta-neutral call-leg cost basis; with stock and call turnover |
| - | `call_correlation_late_otm` | `run_two_assets_no_costs` (as table 1) | `call_correlation_late_otm` | correlation of the NN and Gamma-neutral call positions without costs over all paths, without the late-OTM paths (spot below 98 at any time index from step 25 on) and on those paths only; explains where the network leaves the Gamma-neutral hedge |
| - | `rothert_safeguards.csv` | the same three runs | flags of the region and of the Rothert strategy | how often the safeguards and the undefined-boundary rule were active |

Strategy labels in code and tables are exactly the legend labels: `NN`, `Rothert`, `Gamma-neutral`;
rows always come in the order NN, Rothert, Gamma-neutral. The Gamma-neutral hedge clips lambda to +-20.
Delta-only and fixed-lambda hedges are not part of this repository. In the TeX versions the NN/Rothert
ratio is printed in the NN row only.

## Training

`scripts/train.py` is one linear script like `scripts/figures.py`: the training paths (100 000 paths, seed 111) and the
market data come first, then one section per network in thesis order. Each section sets its number of
epochs, batch size, learning rate, loss and seeds, calls `torch.manual_seed` right before it creates the
network (initial weights), draws the mini-batches from its own seeded `torch.Generator`, runs the Adam
training loop and saves weights, fingerprint (the dict of training settings written next to the loop) and
loss history with `save_model`. The `kappa_sweep` section trains its five networks in one loop over kappa
with the same seeds; `fine_grid_120` simulates its own 120-step training paths. `two_assets_no_costs` was
trained without seeds: its section draws the mini-batches from the global torch generator, so a retrain
gives other weights than the shipped file and cannot reproduce it bit for bit.

`tests/test_deep_hedging.py` runs `scripts/train.py` with 2 epochs per network in a temporary folder and checks
that every fingerprint it writes equals the one stored in the shipped checkpoint (apart from the number of
epochs), that `load_model` reads every file, and that the first losses of `delta_neutral_basis` equal the
loss history stored in its shipped checkpoint; a second test checks that `scripts/train.py` refuses
`OUT_DIR = "models"`.

## Modules

| module | content |
|---|---|
| `src/config.py` | market parameters, seeds, path counts, the risk aversion and the settings of the Gamma-neutral hedge and of the cross-check |
| `src/black_scholes.py` | Black-Scholes paths, prices, Greeks (including speed) and the market data (payoff, premium, Greeks of target and hedge call) |
| `src/cost_functions.py` | proportional costs, the delta-equivalent cost basis of the call leg, the torch bridge for numpy strategies |
| `src/deep_hedging.py` | the four network classes, P&L functions, losses, `save_model` (used by `scripts/train.py`) and `load_model` (used by `scripts/figures.py` and `scripts/tables.py`) |
| `src/benchmarks.py` | Gamma-neutral hedge, Black-Scholes delta hedge, unhedged position, delta-neutral lines and position tables |
| `src/rothert.py` | the no-trade region and the strategy trading on its boundary, with the page reference of every formula |
| `src/hedging_runs.py` | the strategy runs behind `scripts/figures.py` and `scripts/tables.py`: explicit arguments, numpy arrays back, plus `print_run_summary` |
| `src/evaluation.py` | metrics (`strategy_metrics` of one strategy of a run), `save_table` with the TeX writer, hourglass panels, heatmap data, cross-check |
| `src/plotting.py` | style, colours, the `plot_*` functions (each returns a figure) and `save_figure` |

## Networks

All networks are trained on 100 000 paths (seed 111), batch size 256, with Adam. "Section" is the
section name in `scripts/train.py` and its entry in `TRAIN`. "Used by" refers to the figures and tables above.

| section | checkpoint | architecture | loss | epochs | learning rate | seeds (weights, batches) | used by |
|---|---|---|---|---|---|---|---|
| `single_asset_no_costs` | `hedgenet_nb01_entropic_l1.pt` | feed-forward, 2 inputs, 32 hidden | entropic, lambda 1 | 15000 | 5e-4 | 20260919, 20260920 | figure 1, table 1 |
| `two_assets_no_costs` | `hedgenet_nb07_mse_3feat.pt` | feed-forward, 3 inputs, 64 hidden, two outputs | mean squared P&L | 15000 | 1e-3 | none (unseeded) | figures 2-3, table 1 |
| `single_asset_costs` | `hedgenet_nb05_entropic_kappa1e-02.pt` | recurrent, 2 inputs + previous delta, 32 hidden, kappa 0.01 | entropic, lambda 1 | 5000 | 1e-3 | 20260922, 20260923 | figure 4 |
| `kappa_sweep`, kappa 0.01 | `hedgenet_kappa_sweep_1e-02.pt` | recurrent two-asset, 3 inputs + 2 previous positions, 64 hidden | entropic, lambda 2 | 10000 | 1e-3 | 20260810, 20260811 | baseline scenario (figures 5-11, tables 2-4), sweep |
| `kappa_sweep`, kappa 3e-3 ... 1e-4 | `hedgenet_kappa_sweep_3e-03.pt`, `_1e-03`, `_3e-04`, `_1e-04` | same, trained at the kappa in the file name | entropic, lambda 2 | 10000 | 1e-3 | 20260810, 20260811 | figures 8-9, table 3 |
| `fine_grid_120` | `hedgenet_nb09_n120_kappa0.01.pt` | same, 120 trading steps | entropic, lambda 2 | 10000 | 1e-3 | 20260810, 20260811 | figure 10, table 4 |
| `delta_neutral_basis` | `hedgenet_nb10_n30_kappa0.01_deltaneutral.pt` | same, call-leg costs on delta_hedge * S | entropic, lambda 2 | 10000 | 1e-3 | 20260810, 20260811 | figure 11, table 4 |

The network of the baseline scenario (kappa = 0.01, 30 steps) is the kappa = 0.01 network of
`kappa_sweep`: the two checkpoints that trained it were weight-identical, so only one file is shipped.

## Rothert's formulas

Source: N. Rothert, "Optimizing A Nearly Gamma-neutral Portfolio Under Small Transaction Costs",
Master's thesis, Kiel University (CAU), February 2024. Page numbers are the printed pages (PDF page =
printed page + 6). Every formula in `src/rothert.py` carries a one-line reference directly above its
return expression:

| function | reference |
|---|---|
| `midpoint` | p. 52, eq. (2.7); Rothert's m is replaced by the target option's gamma |
| `to_rothert_coordinates`, `from_rothert_coordinates` | p. 54, eq. (2.9) |
| `c_S` | p. 44 (unnumbered) |
| `c_delta_call` | p. 63 (symbol used, not derived there); derived as Gamma_hedge^2 c_S |
| `c_S_perp` | p. 63 (used); from dS_perp of p. 55, eq. (2.11) with p. 44, eq. (2.3) |
| `c_d_par` | p. 63 (unnumbered); closed form derived for the sold-option setting |
| `alpha_perp` | p. 63 (unnumbered) |
| `beta_bar_perp` | p. 69 (unnumbered) |
| `alpha_par` | p. 63 (definition) and pp. 66-69; closed form derived here |
| `beta_par` | p. 67 (unnumbered) |
| `h_par` | p. 68, eq. (2.28) |
| `h_perp` | p. 69 (final boundary, unnumbered); printed exponent eps^(2/7) |
| `inside_region` | pp. 68-69 |
| `project_to_boundary` | p. 65 (indicator structure); parallel-then-perpendicular is our discretisation |

The boundaries contain no calibrated parameter: epsilon is the cost rate kappa and the risk aversion is
2, both set by the problem, and nothing is tuned on paths.

## Safeguards

`gamma_floor` (a lower bound on the hedge call's gamma) and the position caps are numerical safeguards
of this project and not part of Rothert's thesis. They are off by default. If switched on, every
affected observation is flagged (`gamma_floor_active`, `stock_cap_active`, `call_cap_active`), and
`tables/rothert_safeguards.csv` reports how often each flag was active together with
`gamma_problematic` (hedge gamma below 1e-6, a diagnostic only) and `region_undefined`. An undefined
region boundary (non-finite boundary or position) counts as inside, so no trade happens there; this is
an observation that is reported, not a safeguard.

## Reproducibility

- Test paths use seed 999, training paths seed 111, both with the global numpy generator.
- Everything runs on the CPU. Results were produced with torch 2.11.0 and 8 threads; other torch versions
  or thread counts can change the last bits of the floating point results.
- Seeded networks retrain bit-identically on the machine they were produced on: the first 30 epochs of all
  ten networks from `scripts/train.py` equal those of the original training code, and a full retrain of
  `single_asset_costs` (5000 epochs) reproduces the shipped weights exactly; `two_assets_no_costs` was
  trained without seeds and cannot be reproduced bit for bit.
- A retrained checkpoint stores the number of epochs actually trained in its fingerprint, so a shortened
  run (`N_EPOCHS_OVERRIDE`) can be told apart from a full one; `scripts/figures.py` and `scripts/tables.py` load it with
  `load_model` anyway (for inspection only).
- The `pnl_fn` and `loss_name` strings and the key names inside the fingerprints are historical
  identifiers of the original training code; `scripts/train.py` writes them unchanged so that its fingerprints
  equal the shipped ones.
