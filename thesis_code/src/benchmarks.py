"""Analytic benchmarks: Gamma-neutral hedge, Black-Scholes delta hedge, unhedged position."""

import numpy as np
import pandas as pd

from src.black_scholes import bs_call_delta
from src.cost_functions import apply_cost_numpy


def gamma_neutral_strategy(paths, market, clip_lambda, eps=1e-8, cost_fn=None, cost_fn_call=None,
                           call_cost_basis=None):
    """Delta- and gamma-neutral hedge, rebalanced at every step.

    Per step: lambda = gamma_target / (gamma_hedge + eps), clipped to +-clip_lambda; stock position
    delta_target - lambda * delta_hedge. Without cost_fn no costs are charged. With cost_fn both legs
    pay it (the call leg cost_fn_call if given, on call_cost_basis instead of the call price if given)
    and both positions are liquidated at maturity.

    Returns (pnl, stock_positions, call_positions, total_costs); call positions are lambda.
    """
    n_paths, n_times = paths.shape
    n_steps = n_times - 1

    call_hedge_prices = market.call_hedge
    payoff = market.payoff
    p0 = market.p0

    hedge_gain          = np.zeros(n_paths)
    total_costs         = np.zeros(n_paths)
    delta_stock_history = np.zeros((n_paths, n_steps))
    lambda_history      = np.zeros((n_paths, n_steps))

    if cost_fn is None:
        for k in range(n_steps):
            S_k    = paths[:, k]
            S_next = paths[:, k + 1]
            C_k    = call_hedge_prices[:, k]
            C_next = call_hedge_prices[:, k + 1]

            lam_k = market.gamma_target[:, k] / (market.gamma_hedge[:, k] + eps)
            lam_k = np.clip(lam_k, -clip_lambda, clip_lambda)

            delta_stock_k = market.delta_target[:, k] - lam_k * market.delta_hedge[:, k]

            hedge_gain += delta_stock_k * (S_next - S_k) + lam_k * (C_next - C_k)

            delta_stock_history[:, k] = delta_stock_k
            lambda_history[:, k]      = lam_k

        pnl = p0 - payoff + hedge_gain
        return pnl, delta_stock_history, lambda_history, total_costs

    cost_fn_call_used = cost_fn if cost_fn_call is None else cost_fn_call
    if call_cost_basis is not None and tuple(call_cost_basis.shape) != tuple(call_hedge_prices.shape):
        raise ValueError(
            f"call_cost_basis.shape {tuple(call_cost_basis.shape)} != "
            f"call_hedge.shape {tuple(call_hedge_prices.shape)}"
        )

    delta_stock_prev = np.zeros(n_paths)
    lambda_prev      = np.zeros(n_paths)

    for k in range(n_steps):
        S_k    = paths[:, k]
        S_next = paths[:, k + 1]
        C_k    = call_hedge_prices[:, k]
        C_next = call_hedge_prices[:, k + 1]

        lam_k = market.gamma_target[:, k] / (market.gamma_hedge[:, k] + eps)
        lam_k = np.clip(lam_k, -clip_lambda, clip_lambda)

        delta_stock_k = market.delta_target[:, k] - lam_k * market.delta_hedge[:, k]

        C_k_for_cost = C_k if call_cost_basis is None else call_cost_basis[:, k]
        total_costs += apply_cost_numpy(cost_fn, delta_stock_k, delta_stock_prev, S_k)
        total_costs += apply_cost_numpy(cost_fn_call_used, lam_k, lambda_prev, C_k_for_cost)

        hedge_gain += delta_stock_k * (S_next - S_k) + lam_k * (C_next - C_k)

        delta_stock_history[:, k] = delta_stock_k
        lambda_history[:, k]      = lam_k
        delta_stock_prev          = delta_stock_k
        lambda_prev               = lam_k

    S_T  = paths[:, -1]
    C_T  = call_hedge_prices[:, -1]
    C_T_for_cost = C_T if call_cost_basis is None else call_cost_basis[:, -1]
    zero = np.zeros(n_paths)
    total_costs += apply_cost_numpy(cost_fn, zero, delta_stock_prev, S_T)
    total_costs += apply_cost_numpy(cost_fn_call_used, zero, lambda_prev, C_T_for_cost)

    pnl = p0 - payoff + hedge_gain - total_costs
    return pnl, delta_stock_history, lambda_history, total_costs


def delta_hedge_pnl(paths, payoff, p0, K, T_total, r, sigma):
    """P&L of the Black-Scholes delta hedge without costs. Returns (pnl, deltas)."""
    n_steps = paths.shape[1] - 1

    deltas = np.zeros((paths.shape[0], n_steps))
    for k in range(n_steps):
        tau = T_total - k * T_total / n_steps
        deltas[:, k] = bs_call_delta(S=paths[:, k], K=K, T=tau, r=r, sigma=sigma)

    dS = paths[:, 1:] - paths[:, :-1]
    hedge_gain = np.sum(deltas * dS, axis=1)
    pnl = p0 - payoff + hedge_gain
    return pnl, deltas


def unhedged_pnl(payoff, p0):
    """P&L of the sold option without any hedge."""
    return -payoff + p0


def delta_neutral_line(delta_target, delta_hedge, stock_grid, eps=1e-12):
    """Delta-neutral line in position space: phi_call = (delta_target - phi_stock) / delta_hedge.

    NaN if |delta_hedge| < eps (vertical line).
    """
    stock_grid = np.asarray(stock_grid, dtype=float)
    delta_target = float(delta_target)
    delta_hedge = float(delta_hedge)

    if abs(delta_hedge) < eps:
        return np.full_like(stock_grid, np.nan)

    return (delta_target - stock_grid) / delta_hedge


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def single_path_positions(path_idx, steps, nn_stock, nn_call, gn_stock, gn_call, delta_target,
                          delta_hedge):
    """Positions of NN and Gamma-neutral hedge and the option deltas for one path at the given steps.

    Position matrices are (n_paths, n_steps); delta matrices may have one more column and are cut.
    """
    nn_s = _to_numpy(nn_stock)
    nn_c = _to_numpy(nn_call)
    gn_s = _to_numpy(gn_stock)
    gn_c = _to_numpy(gn_call)
    dt = _to_numpy(delta_target)
    dh = _to_numpy(delta_hedge)

    n_paths, n_steps = nn_s.shape

    if not (0 <= path_idx < n_paths):
        raise ValueError(f"path_idx={path_idx} outside the valid range [0, {n_paths - 1}]")

    dt, dh = dt[:, :n_steps], dh[:, :n_steps]

    rows = []
    for step in steps:
        step = int(step)
        if not (0 <= step < n_steps):
            raise ValueError(f"step={step} outside the valid range [0, {n_steps - 1}]")
        rows.append({
            "step":         step,
            "nn_stock":     float(nn_s[path_idx, step]),
            "nn_call":      float(nn_c[path_idx, step]),
            "gn_stock":     float(gn_s[path_idx, step]),
            "gn_call":      float(gn_c[path_idx, step]),
            "delta_target": float(dt[path_idx, step]),
            "delta_hedge":  float(dh[path_idx, step]),
        })

    return pd.DataFrame(rows)
