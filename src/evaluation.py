"""Metrics, table output and figure data derived from the strategy runs.

The helpers take plain arrays or the flat result dict of hedging_runs.run_with_costs
(strategy_metrics). save_table writes a table as CSV and as a bare booktabs TeX tabular.
"""

import pathlib

import numpy as np
import torch

from src import config
from src.deep_hedging import entropic_loss, pnl_two_assets_with_costs
from src.rothert import (
    from_rothert_coordinates, h_perp, inside_region, to_rothert_coordinates,
)

NO_CAP_NOTE = "n/a (no position limits)"


def rho_lambda(pnl, risk_aversion):
    """Entropic risk (1/lambda) log E[exp(-lambda P&L)] of a terminal P&L distribution, in float64."""
    pnl_t = torch.tensor(pnl, dtype=torch.float64)
    return entropic_loss(pnl_t, lam=risk_aversion).item()


def pnl_statistics(pnl):
    """Nine summary statistics of a terminal P&L distribution (float64, linear quantiles)."""
    pnl64 = np.asarray(pnl, dtype=np.float64)
    return {
        "mean": float(pnl64.mean()),
        "std": float(pnl64.std()),
        "median": float(np.median(pnl64)),
        "min": float(pnl64.min()),
        "max": float(pnl64.max()),
        "q01": float(np.quantile(pnl64, 0.01)),
        "q05": float(np.quantile(pnl64, 0.05)),
        "q95": float(np.quantile(pnl64, 0.95)),
        "mean_abs": float(np.abs(pnl64).mean()),
    }


def cost_statistics(costs):
    """Five summary statistics of the transaction costs per path."""
    costs64 = np.asarray(costs, dtype=np.float64)
    return {
        "mean": float(costs64.mean()),
        "median": float(np.median(costs64)),
        "std": float(costs64.std()),
        "q95": float(np.quantile(costs64, 0.95)),
        "max": float(costs64.max()),
    }


def count_outside(values, window):
    """(number below, number above) the window; boundary values count as inside."""
    values = np.asarray(values, dtype=np.float64)
    lo, hi = window
    return int((values < lo).sum()), int((values > hi).sum())


def call_correlation_late_otm(paths, call_nn, call_gn, step_from, spot_threshold):
    """Correlation of the NN and Gamma-neutral call positions with and without the late-OTM paths.

    A path counts as late-OTM if its spot falls below spot_threshold at any time index from step_from
    on (maturity included). The correlation is taken over all (path, step) pairs of a subset.
    Returns three rows: all paths, without the late-OTM paths, late-OTM paths only.
    """
    late_otm = paths[:, step_from:].min(axis=1) < spot_threshold
    rows = []
    for subset, mask in [
        ("All paths", np.ones(len(paths), dtype=bool)),
        ("Without late-OTM paths", ~late_otm),
        ("Late-OTM paths only", late_otm),
    ]:
        nn = np.asarray(call_nn, dtype=np.float64)[mask].ravel()
        gn = np.asarray(call_gn, dtype=np.float64)[mask].ravel()
        rows.append({
            "subset": subset, "n_paths": int(mask.sum()), "share_paths": float(mask.mean()),
            "corr_call": float(np.corrcoef(nn, gn)[0, 1]),
        })
    return rows


class RothertPositionAdapter:
    """Stands in for the network in the P&L chain and returns the precomputed Rothert position of each step."""

    def __init__(self, phi_stock, phi_call):
        self.phi_stock = torch.tensor(phi_stock, dtype=torch.float32)
        self.phi_call = torch.tensor(phi_call, dtype=torch.float32)
        self.k = 0

    def __call__(self, market_features, delta_stock_prev, delta_call_prev):
        position = torch.stack([self.phi_stock[:, self.k], self.phi_call[:, self.k]], dim=1)
        self.k += 1
        return position


def rothert_cross_check(rothert, tensors, cost_args, rtol, atol):
    """Re-price the Rothert positions through the network P&L chain.

    tensors: dict with float32 tensors paths, call_prices, payoff, p0. cost_args: the keyword
    arguments the network's chain uses for costs. Raises RuntimeError if P&L or costs deviate
    beyond rtol/atol; returns the largest absolute P&L deviation.
    """
    adapter = RothertPositionAdapter(rothert.phi_stock, rothert.phi_call)
    with torch.no_grad():
        pnl_ad, _, _, costs_ad = pnl_two_assets_with_costs(
            adapter, tensors["paths"], tensors["call_prices"], tensors["payoff"], tensors["p0"],
            config.T_TOTAL, config.K_TARGET, **cost_args,
        )
    pnl_ad = pnl_ad.cpu().numpy()
    costs_ad = costs_ad.cpu().numpy()
    max_abs = float(np.max(np.abs(pnl_ad - rothert.pnl)))
    if not (
        np.allclose(pnl_ad, rothert.pnl, rtol=rtol, atol=atol)
        and np.allclose(costs_ad, rothert.total_costs, rtol=rtol, atol=atol)
    ):
        raise RuntimeError(
            f"cross-check failed: largest absolute P&L deviation {max_abs:.2e} "
            f"(rtol={rtol:.0e}, atol={atol:.0e})"
        )
    return max_abs


def _turnover(position):
    """Absolute position changes per path including the initial and the liquidation trade."""
    parts = [np.abs(position[:, :1]), np.abs(np.diff(position, axis=1)), np.abs(position[:, -1:])]
    return np.concatenate(parts, axis=1).sum(axis=1)


def trade_statistics(stock, call, traded=None, stock_cap_active=None, call_cap_active=None):
    """Seven trading statistics of a strategy (positions (n_paths, n_steps)).

    traded=None derives the trade flag exactly (any position change, no tolerance). Without cap
    arrays the cap columns carry NO_CAP_NOTE, because a safeguard that does not exist is not zero.
    """
    stock = np.asarray(stock, dtype=np.float64)
    call = np.asarray(call, dtype=np.float64)

    if traded is None:
        stock_changed = np.diff(stock, axis=1, prepend=0.0) != 0
        call_changed = np.diff(call, axis=1, prepend=0.0) != 0
        traded = stock_changed | call_changed
    else:
        traded = np.asarray(traded, dtype=bool)

    if stock_cap_active is None:
        pct_stock_cap_active = NO_CAP_NOTE
    else:
        pct_stock_cap_active = float(np.asarray(stock_cap_active, dtype=bool).mean())

    if call_cap_active is None:
        pct_call_cap_active = NO_CAP_NOTE
    else:
        pct_call_cap_active = float(np.asarray(call_cap_active, dtype=bool).mean())

    return {
        "mean_stock_turnover": float(_turnover(stock).mean()),
        "mean_call_turnover": float(_turnover(call).mean()),
        "mean_n_trades": float(traded.sum(axis=1).mean()),
        "pct_steps_no_trade": float((~traded).mean()),
        "pct_paths_no_rebalancing": float((traded.sum(axis=1) == 0).mean()),
        "pct_stock_cap_active": pct_stock_cap_active,
        "pct_call_cap_active": pct_call_cap_active,
    }


def strategy_metrics(run, short):
    """All metrics of one strategy of a run_with_costs result; short is "nn", "rothert" or "gn".

    Keys: pnl_* (nine P&L statistics), cost_* (five cost statistics), the seven trade statistics, rho.
    Only Rothert has its own trade flags and position caps. For NN and Gamma-neutral a trade is any
    position change and the cap columns read NO_CAP_NOTE.
    """
    pnl = run[f"pnl_{short}"]
    row = {f"pnl_{k}": v for k, v in pnl_statistics(pnl).items()}
    row.update({f"cost_{k}": v for k, v in cost_statistics(run[f"costs_{short}"]).items()})
    if short == "rothert":
        row.update(trade_statistics(
            run["stock_rothert"], run["call_rothert"], traded=run["traded_rothert"],
            stock_cap_active=run["stock_cap_active"], call_cap_active=run["call_cap_active"],
        ))
    else:
        row.update(trade_statistics(run[f"stock_{short}"], run[f"call_{short}"]))
    row["rho"] = rho_lambda(pnl, config.RISK_AVERSION)
    return row


def _tex_escape(text):
    for char in ("&", "%", "_", "#"):
        text = text.replace(char, "\\" + char)
    return text


def latex_tabular(df, columns):
    """Bare booktabs tabular. columns: list of (header, column, format); format None means text.

    Headers are LaTeX as given, numeric cells are set in math mode so minus signs typeset.
    """
    alignment = "".join("l" if fmt is None else "r" for _, _, fmt in columns)
    lines = ["\\begin{tabular}{" + alignment + "}"]
    lines.append("\\toprule")
    lines.append(" & ".join(header for header, _, _ in columns) + " \\\\")
    lines.append("\\midrule")
    for _, row in df.iterrows():
        cells = []
        for _, column, fmt in columns:
            value = row[column]
            if fmt is None:
                cells.append(_tex_escape(str(value)))
            elif value is None or (isinstance(value, float) and np.isnan(value)):
                cells.append("")
            else:
                cells.append("$" + format(value, fmt) + "$")
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def save_table(df, name, tex_columns=None, out_dir="tables", tex_frame=None):
    """Write df as out_dir/name.csv and, if tex_columns is given, as out_dir/name.tex.

    The TeX file is built from tex_frame if given, else from df (see latex_tabular).
    Returns the list of written paths.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    df.to_csv(csv_path, index=False)
    if tex_columns is None:
        print(f"saved {csv_path}")
        return [csv_path]
    frame = df if tex_frame is None else tex_frame
    tex_path = out_dir / f"{name}.tex"
    tex_path.write_text(latex_tabular(frame, tex_columns), encoding="utf-8")
    print(f"saved {csv_path} and {tex_path.name}")
    return [csv_path, tex_path]


def select_median_path(traded):
    """Path whose number of trades is closest to the median over all paths. Returns (index, n_trades).

    traded: boolean array (n_paths, n_steps), True where the strategy trades.
    """
    n_trades = traded.sum(axis=1)
    idx = int(np.argmin(np.abs(n_trades - np.median(n_trades))))
    return idx, int(n_trades[idx])


def select_panel_steps(traded, path_idx, n_panels):
    """Up to n_panels trading steps of one path, evenly spread over its trades (step 0 not alone)."""
    traded_steps = np.flatnonzero(traded[path_idx])
    n_traded = int(traded_steps.size)

    if n_traded == 0:
        return []

    if n_traded <= n_panels:
        steps = sorted(int(k) for k in traded_steps)
    else:
        positions = np.unique(
            np.round(np.linspace(0, n_traded - 1, n_panels)).astype(int)
        )
        steps = sorted(int(traded_steps[p]) for p in positions)

        if len(steps) == 1 and steps[0] == 0:
            after_zero = traded_steps[traded_steps > 0]
            if after_zero.size > 0:
                steps = sorted({0, int(after_zero[0])})

    return steps


def hourglass_boundary(h_par_k, alpha_perp_k, beta_bar_perp_k, epsilon, delta_hedge_k, n_points):
    """Boundary of the region in cartesian position deviations, relative to the midpoint.

    Returns (upper half, lower half), each a pair (d_stock, d_call).
    """
    d_par_grid = np.linspace(-h_par_k, h_par_k, n_points)
    h_perp_grid = h_perp(d_par_grid, alpha_perp_k, beta_bar_perp_k, epsilon)

    upper = from_rothert_coordinates(h_perp_grid, d_par_grid, delta_hedge_k)
    lower = from_rothert_coordinates(-h_perp_grid, d_par_grid, delta_hedge_k)

    return upper, lower


def hourglass_panels(region, traded_rothert, stock_rothert, call_rothert, stock_nn, call_nn,
                     epsilon, n_panels, n_points):
    """Median-case path with the region boundary and the positions before and after the trade.

    Positions and traded flags are (n_paths, n_steps) arrays of one run. Returns
    (panels, path index, number of Rothert trades on that path, panel steps). Raises RuntimeError
    unless every panel step is a Rothert trade and step 0 appears at most once.
    """
    idx, n_trades = select_median_path(traded_rothert)
    steps = select_panel_steps(traded_rothert, idx, n_panels)
    if not steps or not all(bool(traded_rothert[idx, k]) for k in steps):
        raise RuntimeError(f"not every panel step {steps} is a trade")
    if steps.count(0) > 1:
        raise RuntimeError(f"step 0 appears more than once in {steps}")

    def _position(stock, call, k):
        # position before the trade at k is the one held after k-1; (0, 0) at k = 0
        if k == 0:
            return (0.0, 0.0)
        return (float(stock[idx, k - 1]), float(call[idx, k - 1]))

    panels = []
    for k in steps:
        h_par = float(region.h_par[idx, k])
        mid = (float(region.phi_stock_star[idx, k]), float(region.phi_call_star[idx, k]))
        upper, lower = hourglass_boundary(
            h_par_k=h_par,
            alpha_perp_k=float(region.alpha_perp[idx, k]),
            beta_bar_perp_k=float(region.beta_bar_perp[idx, k]),
            epsilon=epsilon,
            delta_hedge_k=float(region.market.delta_hedge[idx, k]),
            n_points=n_points,
        )
        # closed polyline: upper half forward, lower half backward, start point repeated
        boundary_stock = np.concatenate([upper[0], lower[0][::-1], upper[0][:1]]) + mid[0]
        boundary_call = np.concatenate([upper[1], lower[1][::-1], upper[1][:1]]) + mid[1]
        panels.append({
            "k": k, "h_par": h_par, "midpoint": mid,
            "boundary_stock": boundary_stock, "boundary_call": boundary_call,
            "rothert_before": _position(stock_rothert, call_rothert, k),
            "rothert_after": (float(stock_rothert[idx, k]), float(call_rothert[idx, k])),
            "nn_before": _position(stock_nn, call_nn, k),
            "nn_after": (float(stock_nn[idx, k]), float(call_nn[idx, k])),
        })
    return panels, idx, n_trades, steps


def classify_positions(region, paths, stock, call, epsilon):
    """Per observation: is the position held before the step inside the region and defined?

    stock, call: positions (n_paths, n_steps). Returns (inside_and_defined, undefined).
    """
    n_paths, n_times = paths.shape
    n_steps = n_times - 1

    stock = np.asarray(stock, dtype=np.float64)
    call = np.asarray(call, dtype=np.float64)
    expected_shape = (n_paths, n_steps)
    if stock.shape != expected_shape or call.shape != expected_shape:
        raise ValueError(
            f"stock/call must have shape {expected_shape}, got {stock.shape} and {call.shape}"
        )

    phi_stock_prev = np.zeros(n_paths)
    phi_call_prev = np.zeros(n_paths)

    inside_hist = np.zeros((n_paths, n_steps), dtype=bool)
    undefined_hist = np.zeros((n_paths, n_steps), dtype=bool)

    for k in range(n_steps):
        d1 = phi_stock_prev - region.phi_stock_star[:, k]
        d2 = phi_call_prev - region.phi_call_star[:, k]

        d_perp_k, d_par_k = to_rothert_coordinates(d1, d2, region.market.delta_hedge[:, k])
        inside_k, undefined_k = inside_region(
            d_perp_k, d_par_k, region.h_par[:, k],
            region.alpha_perp[:, k], region.beta_bar_perp[:, k], epsilon,
        )
        inside_hist[:, k] = inside_k
        undefined_hist[:, k] = undefined_k

        phi_stock_prev = stock[:, k]
        phi_call_prev = call[:, k]

    return inside_hist & ~undefined_hist, undefined_hist


def normalized_moves(region, paths, stock, call, epsilon):
    """Positions before and after each step in the normalised coordinates of the region.

    x = d_par / h_par, y = d_perp / h_perp(h_par). The hourglass is |x| <= 1, |y| <= |x|^(2/3).
    Returns x_before, y_before, gamma_move = |dx|, delta_move = |dy| and hourglass_inside.
    """
    n_paths, n_times = paths.shape
    n_steps = n_times - 1

    stock = np.asarray(stock, dtype=np.float64)
    call = np.asarray(call, dtype=np.float64)
    expected_shape = (n_paths, n_steps)
    if stock.shape != expected_shape or call.shape != expected_shape:
        raise ValueError(
            f"stock/call must have shape {expected_shape}, got {stock.shape} and {call.shape}"
        )

    stock_before = np.concatenate([np.zeros((n_paths, 1)), stock[:, :-1]], axis=1)
    call_before = np.concatenate([np.zeros((n_paths, 1)), call[:, :-1]], axis=1)

    phi_stock_star_k = region.phi_stock_star[:, :n_steps]
    phi_call_star_k = region.phi_call_star[:, :n_steps]
    delta_hedge_k = region.market.delta_hedge[:, :n_steps]
    h_par_k = region.h_par[:, :n_steps]
    alpha_perp_k = region.alpha_perp[:, :n_steps]
    beta_bar_perp_k = region.beta_bar_perp[:, :n_steps]

    d_perp_before, d_par_before = to_rothert_coordinates(
        stock_before - phi_stock_star_k, call_before - phi_call_star_k, delta_hedge_k
    )
    d_perp_after, d_par_after = to_rothert_coordinates(
        stock - phi_stock_star_k, call - phi_call_star_k, delta_hedge_k
    )

    scale_perp = h_perp(h_par_k, alpha_perp_k, beta_bar_perp_k, epsilon)

    with np.errstate(divide="ignore", invalid="ignore"):
        x_before = d_par_before / h_par_k
        y_before = d_perp_before / scale_perp
        x_after = d_par_after / h_par_k
        y_after = d_perp_after / scale_perp

    gamma_move = np.abs(x_after - x_before)
    delta_move = np.abs(y_after - y_before)
    hourglass_inside = (np.abs(x_before) <= 1.0) & (np.abs(y_before) <= np.abs(x_before) ** (2.0 / 3.0))

    return {
        "x_before": x_before,
        "y_before": y_before,
        "gamma_move": gamma_move,
        "delta_move": delta_move,
        "hourglass_inside": hourglass_inside,
    }


def bin_heatmap(x, y, values, window, n_bins, min_count):
    """Mean of values in a square grid over [-window, window]^2; cells with fewer than min_count are NaN.

    Index order [x cell, y cell]. Returns edges, counts and mean.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    values = np.asarray(values, dtype=np.float64).ravel()

    edges = np.linspace(-window, window, n_bins + 1)
    in_window = (
        np.isfinite(x) & np.isfinite(y) & (np.abs(x) <= window) & (np.abs(y) <= window)
    )
    if not np.all(np.isfinite(values[in_window])):
        raise ValueError("values contain non-finite entries inside the window; nothing is filtered silently")

    counts, _, _ = np.histogram2d(x[in_window], y[in_window], bins=[edges, edges])
    counts = counts.astype(np.int64)
    total, _, _ = np.histogram2d(
        x[in_window], y[in_window], bins=[edges, edges], weights=values[in_window]
    )
    mean = np.where(counts >= min_count, total / np.maximum(counts, 1), np.nan)

    return {"edges": edges, "counts": counts, "mean": mean}


def gamma_move_map(region, paths, stock, call, epsilon, window, n_bins, min_count,
                   check_no_trade_inside=False):
    """Mean gamma move per cell of the normalised position plane for one strategy at cost rate epsilon.

    stock, call: positions (n_paths, n_steps) of the run that belongs to region. Steps k >= 1 only.
    Raises RuntimeError if the hourglass test deviates from the region test anywhere, or, with
    check_no_trade_inside=True (Rothert), if the strategy moves from inside the region.
    Returns (mean map with index [x cell, y cell], bin edges).
    """
    inside_and_defined, _ = classify_positions(region, paths, stock, call, epsilon)
    moves = normalized_moves(region, paths, stock, call, epsilon)

    x_sel = moves["x_before"][:, 1:]
    y_sel = moves["y_before"][:, 1:]
    inside_sel = inside_and_defined[:, 1:]

    deviations = int((moves["hourglass_inside"][:, 1:] != inside_sel).sum())
    if deviations != 0:
        raise RuntimeError(
            f"hourglass test deviates from the region test at {deviations} observations "
            f"(epsilon={epsilon:g})"
        )
    if check_no_trade_inside:
        max_gamma = float(moves["gamma_move"][:, 1:][inside_sel].max(initial=0.0))
        max_delta = float(moves["delta_move"][:, 1:][inside_sel].max(initial=0.0))
        if max_gamma != 0.0 or max_delta != 0.0:
            raise RuntimeError(
                f"strategy moves from inside the region (epsilon={epsilon:g}): "
                f"max gamma move {max_gamma}, max delta move {max_delta}"
            )

    binned = bin_heatmap(x_sel, y_sel, moves["gamma_move"][:, 1:], window, n_bins, min_count)
    return binned["mean"], binned["edges"]
