"""Strategy runs behind the figures in scripts/figures.py and the tables in scripts/tables.py.

Each function takes the test paths, the market data and a loaded network and returns numpy arrays.
Market parameters come from src/config.py. Nothing is trained here.
"""

from functools import partial

import numpy as np
import torch

from src import config
from src.benchmarks import delta_hedge_pnl, gamma_neutral_strategy, unhedged_pnl
from src.black_scholes import bs_call_delta, bs_call_price, call_payoff
from src.cost_functions import delta_equivalent_call_cost_basis, proportional_cost
from src.deep_hedging import (
    pnl_single_asset, pnl_single_asset_with_costs, pnl_two_assets, pnl_two_assets_with_costs,
)
from src.evaluation import (
    cost_statistics, count_outside, pnl_statistics, rho_lambda, rothert_cross_check,
)
from src.rothert import RothertConfig, compute_rothert_region, run_rothert_strategy


def _tensor(x):
    return torch.tensor(x, dtype=torch.float32)


def run_with_costs(paths, market, model, kappa, cost_model="proportional"):
    """NN, Rothert and Gamma-neutral hedge under proportional costs kappa on the same paths.

    cost_model "proportional": both legs pay kappa * price.
    cost_model "delta_neutral": the call leg pays kappa * delta_hedge * S, so one unit of delta
    costs the same through the stock and through the call.

    Returns one flat dict:
        pnl_nn, pnl_rothert, pnl_gn              terminal P&L per path
        costs_nn, costs_rothert, costs_gn        total costs per path
        stock_nn, call_nn, stock_rothert, call_rothert, stock_gn, call_gn   positions (n_paths, n_steps)
        traded_rothert                           True where Rothert trades (n_paths, n_steps)
        region_undefined                         True where the region boundary is undefined
        stock_cap_active, call_cap_active        Rothert position caps hit (all False: caps are off)
        region                                   the RothertRegion object
        cross_check_max_abs                      largest P&L deviation of the Rothert cross-check
    """
    paths_t = _tensor(paths)
    call_prices_t = _tensor(market.call_hedge)
    payoff_t = _tensor(market.payoff)
    p0_t = _tensor(market.p0)

    cost_fn = partial(proportional_cost, c=kappa)
    if cost_model == "proportional":
        cost_fn_call = None  # the call leg uses cost_fn on the call price
        basis = None
        basis_t = None
    elif cost_model == "delta_neutral":
        cost_fn_call = partial(proportional_cost, c=kappa)
        basis = delta_equivalent_call_cost_basis(market.delta_hedge, paths)  # delta_hedge * S
        basis_t = _tensor(basis)
    else:
        raise ValueError(f"unknown cost_model {cost_model!r}")

    rothert_config = RothertConfig(
        K_target=config.K_TARGET, K_hedge=config.K_HEDGE, T_total=config.T_TOTAL,
        sigma=config.SIGMA, r=config.R, hedge_maturity_factor=config.HEDGE_MATURITY_FACTOR,
        epsilon=kappa, risk_aversion=config.RISK_AVERSION,
        gamma_floor=None, stock_position_bounds=None, call_position_bounds=None,  # project safeguards, not Rothert's, off
    )
    region = compute_rothert_region(paths, rothert_config, market)

    with torch.no_grad():
        pnl_nn, stock_nn, call_nn, costs_nn = pnl_two_assets_with_costs(
            model=model, paths=paths_t, call_prices=call_prices_t, payoff=payoff_t, p0=p0_t,
            T_total=config.T_TOTAL, K_target=config.K_TARGET, cost_fn=cost_fn,
            cost_fn_call=cost_fn_call, call_cost_basis=basis_t,
        )
    rothert = run_rothert_strategy(
        paths, rothert_config, region, cost_fn=cost_fn, cost_fn_call=cost_fn_call,
        call_cost_basis=basis,
    )
    pnl_gn, stock_gn, call_gn, costs_gn = gamma_neutral_strategy(
        paths, market, config.GN_CLIP_LAMBDA, eps=config.GN_EPS, cost_fn=cost_fn,
        cost_fn_call=cost_fn_call, call_cost_basis=basis,
    )

    # Rothert positions pushed through the network's P&L chain must give the same P&L
    tensors = {"paths": paths_t, "call_prices": call_prices_t, "payoff": payoff_t, "p0": p0_t}
    cost_args = {"cost_fn": cost_fn, "cost_fn_call": cost_fn_call, "call_cost_basis": basis_t}
    cross_check_max_abs = rothert_cross_check(
        rothert, tensors, cost_args, config.CROSS_CHECK_RTOL, config.CROSS_CHECK_ATOL
    )

    return {
        "pnl_nn": pnl_nn.cpu().numpy(), "costs_nn": costs_nn.cpu().numpy(),
        "stock_nn": stock_nn.cpu().numpy(), "call_nn": call_nn.cpu().numpy(),
        "pnl_rothert": rothert.pnl, "costs_rothert": rothert.total_costs,
        "stock_rothert": rothert.phi_stock, "call_rothert": rothert.phi_call,
        "traded_rothert": rothert.traded,
        "pnl_gn": pnl_gn, "costs_gn": costs_gn, "stock_gn": stock_gn, "call_gn": call_gn,
        "region_undefined": rothert.region_undefined,
        "stock_cap_active": rothert.stock_cap_active,
        "call_cap_active": rothert.call_cap_active,
        "region": region,
        "cross_check_max_abs": cross_check_max_abs,
    }


def print_run_summary(label, run, window=None):
    """Print the key numbers and the honesty numbers of one run_with_costs result.

    window = (lo, hi): also print how many paths lie outside the drawn P&L window.
    """
    print(f"[{label}]")
    strategies = (("NN", "nn"), ("Rothert", "rothert"), ("Gamma-neutral", "gn"))
    for name, short in strategies:
        pnl = run[f"pnl_{short}"]
        stat = pnl_statistics(pnl)
        rho = rho_lambda(pnl, config.RISK_AVERSION)
        mean_cost = cost_statistics(run[f"costs_{short}"])["mean"]
        line = (
            f"  {name:13s} P&L mean {stat['mean']:.4f}, std {stat['std']:.4f}, "
            f"rho {rho:.4f}, mean cost {mean_cost:.4f}"
        )
        if window is not None:
            below, above = count_outside(pnl, window)
            line += f"; outside {window}: {below} below, {above} above (not drawn)"
        print(line)

    rho_nn = rho_lambda(run["pnl_nn"], config.RISK_AVERSION)
    rho_rothert = rho_lambda(run["pnl_rothert"], config.RISK_AVERSION)
    print(f"  rho NN / rho Rothert {rho_nn / rho_rothert:.4f}")

    n_undefined = int(run["region_undefined"].sum())
    n_observations = int(run["region_undefined"].size)
    print(
        f"  Rothert: region undefined at {n_undefined} of {n_observations} trading observations; "
        f"gamma floor active {int(run['region'].gamma_floor_active.sum())}, "
        f"stock cap active {int(run['stock_cap_active'].sum())}, "
        f"call cap active {int(run['call_cap_active'].sum())}; "
        f"cross-check max abs deviation {run['cross_check_max_abs']:.2e}"
    )


def run_single_asset_no_costs(paths, model):
    """One-asset network, Black-Scholes delta hedge and unhedged position without costs.

    Returns (pnl_nn, pnl_delta, pnl_unhedged).
    """
    z_test = call_payoff(paths[:, -1], config.K_SINGLE)
    p0 = bs_call_price(S=config.S0, K=config.K_SINGLE, T=config.T_TOTAL, r=config.R, sigma=config.SIGMA)
    with torch.no_grad():
        pnl_t = pnl_single_asset(
            model, paths=_tensor(paths), payoff=_tensor(z_test),
            p0=_tensor(p0), T_total=config.T_TOTAL, K=config.K_SINGLE,
        )
    pnl_nn = pnl_t.cpu().numpy()

    pnl_delta, _ = delta_hedge_pnl(
        paths, z_test, p0, config.K_SINGLE, config.T_TOTAL, config.R, config.SIGMA
    )
    pnl_unhedged = unhedged_pnl(z_test, p0)
    return pnl_nn, pnl_delta, pnl_unhedged


def delta_curve(model, s_grid, k, n_steps):
    """Delta of the one-asset network and Black-Scholes delta over s_grid at step k of n_steps.

    Returns (delta_nn, delta_bs).
    """
    tau = config.T_TOTAL - k * config.T_TOTAL / n_steps
    x = _tensor(
        np.column_stack([
            np.log(s_grid / config.K_SINGLE), np.full_like(s_grid, tau / config.T_TOTAL),
        ])
    )
    with torch.no_grad():
        delta_nn = model(x).squeeze().numpy()
    delta_bs = bs_call_delta(S=s_grid, K=config.K_SINGLE, T=tau, r=config.R, sigma=config.SIGMA)
    return delta_nn, delta_bs


def run_two_assets_no_costs(paths, market, model):
    """Two-asset network and Gamma-neutral hedge without costs.

    Returns (pnl_nn, stock_nn, call_nn, pnl_gn, stock_gn, call_gn).
    """
    with torch.no_grad():
        pnl_t, stock_t, call_t = pnl_two_assets(
            model=model, paths=_tensor(paths), call_prices=_tensor(market.call_hedge),
            payoff=_tensor(market.payoff), p0=_tensor(market.p0), T_total=config.T_TOTAL,
            K_target=config.K_TARGET,
        )
    pnl_gn, stock_gn, call_gn, _ = gamma_neutral_strategy(
        paths, market, config.GN_CLIP_LAMBDA, eps=config.GN_EPS
    )
    return pnl_t.cpu().numpy(), stock_t.cpu().numpy(), call_t.cpu().numpy(), pnl_gn, stock_gn, call_gn


def run_single_asset_with_costs(paths, model, kappa):
    """One-asset recurrent network with proportional costs kappa. Returns (pnl, costs)."""
    z_test = call_payoff(paths[:, -1], config.K_SINGLE)
    p0 = bs_call_price(S=config.S0, K=config.K_SINGLE, T=config.T_TOTAL, r=config.R, sigma=config.SIGMA)
    cost_fn = partial(proportional_cost, c=kappa)
    with torch.no_grad():
        pnl_t, _, costs_t = pnl_single_asset_with_costs(
            model=model, paths=_tensor(paths), payoff=_tensor(z_test),
            p0=_tensor(p0), T_total=config.T_TOTAL, K=config.K_SINGLE, cost_fn=cost_fn,
        )
    return pnl_t.cpu().numpy(), costs_t.cpu().numpy()


def decision_surface(model, s_grid, prev_grid, k, n_steps):
    """New delta of the recurrent one-asset network over stock price and previous delta at step k.

    Returns (ss, dd, zz), all of shape (len(prev_grid), len(s_grid)).
    """
    tau = config.T_TOTAL - k * config.T_TOTAL / n_steps
    ss, dd = np.meshgrid(s_grid, prev_grid)
    market_features = _tensor(
        np.column_stack([
            np.log(ss.flatten() / config.K_SINGLE), np.full_like(ss.flatten(), tau / config.T_TOTAL),
        ])
    )
    prev_delta = _tensor(dd.flatten()[:, None])
    with torch.no_grad():
        zz = model(market_features, prev_delta).squeeze().numpy().reshape(ss.shape)
    return ss, dd, zz
