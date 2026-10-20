"""No-trade region of the delta-gamma-neutral portfolio and the strategy that trades on its boundary.

Formulas follow N. Rothert, "Optimizing A Nearly Gamma-neutral Portfolio Under Small
Transaction Costs", Master's thesis, Kiel University (CAU), February 2024. Page numbers are
printed pages (PDF page = printed page + 6). Safeguards (gamma floor, position caps) are
project additions, not part of the thesis.
"""

from dataclasses import dataclass
from functools import partial

import numpy as np

from src.black_scholes import MarketData, bs_call_speed, prepare_market_data
from src.cost_functions import apply_cost_numpy, proportional_cost


def _real_pow(x, q):
    """Real branch of a fractional power: alpha_perp, beta_bar_perp and alpha_par are negative."""
    return np.sign(x) * np.abs(x) ** q


@dataclass
class RothertConfig:
    """Scalar settings. epsilon is the proportional cost rate (kappa), risk_aversion is Rothert's p.

    gamma_floor and the position bounds are safeguards, off by default. gamma_problematic_threshold
    only marks small hedge gammas in the diagnostics and enters no formula.
    """
    K_target: float
    K_hedge: float
    T_total: float
    sigma: float
    r: float
    hedge_maturity_factor: float
    epsilon: float
    risk_aversion: float
    gamma_floor: float = None
    stock_position_bounds: tuple = None
    call_position_bounds: tuple = None
    gamma_problematic_threshold: float = 1e-6


@dataclass
class RothertRegion:
    """Region ingredients on the time grid, all arrays (n_paths, n_times).

    h_par and alpha_par are NaN or infinite in the last column (c_d_par vanishes at maturity).
    """
    market: MarketData
    gamma_hedge: np.ndarray
    gamma_problematic: np.ndarray
    gamma_floor_active: np.ndarray
    phi_stock_star: np.ndarray
    phi_call_star: np.ndarray
    alpha_perp: np.ndarray
    beta_bar_perp: np.ndarray
    alpha_par: np.ndarray
    beta_par: np.ndarray
    h_par: np.ndarray


@dataclass
class RothertStrategy:
    """Positions, flags and P&L on the trading grid, all arrays (n_paths, n_steps) except the last three."""
    phi_stock: np.ndarray
    phi_call: np.ndarray
    traded: np.ndarray
    inside_region: np.ndarray
    region_undefined: np.ndarray
    d_par_before_trade: np.ndarray
    stock_cap_active: np.ndarray
    call_cap_active: np.ndarray
    hedge_gain: np.ndarray
    total_costs: np.ndarray
    pnl: np.ndarray


def _validate_inputs(paths, config):
    """Raise ValueError for malformed paths or configuration."""
    paths = np.asarray(paths)

    if paths.ndim != 2:
        raise ValueError(f"paths must be two-dimensional (n_paths, n_times), got ndim={paths.ndim}")
    if paths.shape[1] < 2:
        raise ValueError(f"paths needs at least two time points, got n_times={paths.shape[1]}")
    if not np.all(np.isfinite(paths)):
        raise ValueError("paths contains non-finite entries (NaN or Inf)")
    if np.any(paths <= 0):
        raise ValueError("paths contains non-positive prices (<= 0)")

    positive_fields = {
        "sigma": config.sigma,
        "T_total": config.T_total,
        "hedge_maturity_factor": config.hedge_maturity_factor,
        "risk_aversion": config.risk_aversion,
        "K_target": config.K_target,
        "K_hedge": config.K_hedge,
        "gamma_problematic_threshold": config.gamma_problematic_threshold,
    }
    for name, value in positive_fields.items():
        if not (value > 0):
            raise ValueError(f"config.{name} must be > 0, got {value}")

    if config.epsilon < 0:
        raise ValueError(f"config.epsilon must not be negative, got {config.epsilon}")

    finite_scalars = {
        "K_target": config.K_target,
        "K_hedge": config.K_hedge,
        "T_total": config.T_total,
        "sigma": config.sigma,
        "r": config.r,
        "hedge_maturity_factor": config.hedge_maturity_factor,
        "epsilon": config.epsilon,
        "risk_aversion": config.risk_aversion,
        "gamma_problematic_threshold": config.gamma_problematic_threshold,
    }
    for name, value in finite_scalars.items():
        if not np.isfinite(value):
            raise ValueError(f"config.{name} must be finite, got {value}")

    if config.gamma_floor is not None:
        if not np.isfinite(config.gamma_floor):
            raise ValueError(f"config.gamma_floor must be finite, got {config.gamma_floor}")
        if not (config.gamma_floor > 0):
            raise ValueError(f"config.gamma_floor must be > 0, got {config.gamma_floor}")

    for bound_name, bound in (
        ("stock_position_bounds", config.stock_position_bounds),
        ("call_position_bounds", config.call_position_bounds),
    ):
        if bound is not None:
            lower, upper = bound
            if not (np.isfinite(lower) and np.isfinite(upper)):
                raise ValueError(f"config.{bound_name} must consist of finite values, got {bound}")
            if not (lower <= upper):
                raise ValueError(f"config.{bound_name} must satisfy lower <= upper, got {bound}")


def _apply_gamma_floor(gamma_hedge, gamma_floor):
    """Safeguard (not Rothert): raise |gamma_hedge| to gamma_floor. Returns (gamma, active flags)."""
    if gamma_floor is None:
        return gamma_hedge, np.zeros_like(gamma_hedge, dtype=bool)

    gamma_floor_active = np.abs(gamma_hedge) < gamma_floor
    sign = np.sign(gamma_hedge)
    sign = np.where(sign == 0, 1.0, sign)
    gamma_hedge_effective = np.where(gamma_floor_active, sign * gamma_floor, gamma_hedge)
    return gamma_hedge_effective, gamma_floor_active


def _apply_position_caps(position, bounds):
    """Safeguard (not Rothert): clip a position to bounds. Returns (position, active flags)."""
    if bounds is None:
        return position, np.zeros_like(position, dtype=bool)

    lower, upper = bounds
    position_effective = np.clip(position, lower, upper)
    # NaN passes through np.clip unchanged but compares unequal, so it is excluded from the flag
    cap_active = ~np.isnan(position) & (position != position_effective)
    return position_effective, cap_active


def midpoint(delta_target, gamma_target, delta_hedge, gamma_hedge):
    """Delta-gamma-neutral position (phi_stock_star, phi_call_star): phi_call = Gamma_target / Gamma_hedge."""
    # Rothert (2024), p. 52, eq. (2.7); Rothert's m is replaced by the target option's gamma
    phi_call_star = gamma_target / gamma_hedge
    phi_stock_star = delta_target - phi_call_star * delta_hedge
    return phi_stock_star, phi_call_star


def to_rothert_coordinates(d1, d2, delta_call_hedge):
    """Cartesian deviation (d1, d2) to Rothert coordinates (d_perp, d_par)."""
    # Rothert (2024), p. 54, eq. (2.9)
    normalizer = 1.0 + delta_call_hedge**2
    d_perp = (d1 + delta_call_hedge * d2) / normalizer
    d_par = (d2 - delta_call_hedge * d1) / normalizer
    return d_perp, d_par


def from_rothert_coordinates(d_perp, d_par, delta_call_hedge):
    """Rothert coordinates (d_perp, d_par) back to the cartesian deviation (d1, d2)."""
    # Rothert (2024), p. 54, eq. (2.9)
    d1 = d_perp - delta_call_hedge * d_par
    d2 = delta_call_hedge * d_perp + d_par
    return d1, d2


def c_S(S, sigma):
    """Quadratic variation density of the stock, sigma^2 S^2."""
    # Rothert (2024), p. 44 (unnumbered)
    return sigma**2 * S**2


def c_delta_call(gamma_hedge, c_S):
    """Quadratic variation density of the hedge call's delta, Gamma_hedge^2 c_S."""
    # Rothert (2024), p. 63 (symbol used, not derived there); derived here as Gamma_hedge^2 * c_S
    return gamma_hedge**2 * c_S


def c_S_perp(S, sigma, delta_call_hedge):
    """Quadratic variation density of dS_perp = (1 + C'^2) dS, c_S (1 + C'^2)^2."""
    # Rothert (2024), p. 63 (used); from dS_perp of p. 55, eq. (2.11) with p. 44, eq. (2.3)
    increment_factor = 1.0 + delta_call_hedge**2
    return c_S(S, sigma) * increment_factor**2


def _speed_grid(paths, config):
    """Speed of target and hedge call along the time grid, shapes (n_paths, n_times)."""
    n_paths, n_times = paths.shape
    n_steps = n_times - 1
    T_hedge = config.T_total * config.hedge_maturity_factor

    speed_target = np.zeros_like(paths, dtype=float)
    speed_hedge = np.zeros_like(paths, dtype=float)

    for k in range(n_times):
        t_k = k * config.T_total / n_steps
        tau = config.T_total - t_k
        tau_hedge = T_hedge - t_k
        S_k = paths[:, k]

        speed_target[:, k] = bs_call_speed(S_k, config.K_target, tau, config.r, config.sigma)
        speed_hedge[:, k] = bs_call_speed(S_k, config.K_hedge, tau_hedge, config.r, config.sigma)

    return speed_target, speed_hedge


def c_d_par(S, sigma, gamma_target, gamma_hedge, speed_target, speed_hedge):
    """Quadratic variation density of the parallel midpoint coordinate, c_S (d phi_call_star / dS)^2."""
    # Rothert (2024), p. 63 (unnumbered); closed form derived for the sold-option setting
    d_phi_call_star_dS = (
        speed_target * gamma_hedge - gamma_target * speed_hedge
    ) / gamma_hedge**2
    return c_S(S, sigma) * d_phi_call_star_dS**2


def alpha_perp(c_S_perp, c_delta_call, delta_call_hedge, risk_aversion):
    """Perpendicular coefficient alpha_perp, negative by definition."""
    # Rothert (2024), p. 63 (unnumbered)
    return -(risk_aversion**2 * c_S_perp * (1.0 + delta_call_hedge**2) ** 2) / (
        12.0 * c_delta_call
    )


def beta_bar_perp(S, C_hedge, delta_call_hedge, alpha_perp, risk_aversion):
    """Rescaled perpendicular coefficient beta_bar_perp = 1.5 cbrt(p^2 (S + C' C)^2 alpha_perp)."""
    # Rothert (2024), p. 69 (unnumbered)
    linear_combination = S + delta_call_hedge * C_hedge
    return 1.5 * _real_pow(risk_aversion**2 * linear_combination**2 * alpha_perp, 1.0 / 3.0)


def alpha_par(beta_bar_perp, c_delta_call, c_d_par, delta_call_hedge, S, C_hedge, risk_aversion):
    """Parallel coefficient alpha_par, closed form from its definition and the beta_par relation."""
    # Rothert (2024), p. 63 (definition) and pp. 66-69; closed form derived here
    prefactor_definition = beta_bar_perp * c_delta_call / (
        6.0 * c_d_par * (1.0 + delta_call_hedge**2) ** 2
    )
    prefactor_beta_par = 1.5 * (risk_aversion * (delta_call_hedge * S + C_hedge)) ** (2.0 / 3.0)
    return _real_pow(
        prefactor_definition / _real_pow(prefactor_beta_par / 6.0, 1.0 / 3.0),
        9.0 / 7.0,
    )


def beta_par(alpha_par, S, C_hedge, delta_call_hedge, risk_aversion):
    """Rescaled parallel coefficient beta_par = 1.5 cbrt((p (C' S + C))^2 alpha_par)."""
    # Rothert (2024), p. 67 (unnumbered)
    return 1.5 * _real_pow(
        (risk_aversion * (delta_call_hedge * S + C_hedge)) ** 2 * alpha_par, 1.0 / 3.0
    )


def h_par(alpha_par, beta_par, epsilon):
    """Half width of the region along the parallel coordinate, eps^(1/7) sqrt(beta_par / (6 alpha_par))."""
    # Rothert (2024), p. 68, eq. (2.28)
    return epsilon ** (1.0 / 7.0) * np.sqrt(beta_par / (6.0 * alpha_par))


def h_perp(d_par, alpha_perp, beta_bar_perp, epsilon):
    """Half width perpendicular at parallel deviation d_par; exactly 0 at the waist d_par = 0."""
    # Rothert (2024), p. 69 (final boundary, unnumbered); printed exponent eps^(2/7)
    return (
        epsilon ** (2.0 / 7.0)
        * np.abs(d_par) ** (2.0 / 3.0)
        * np.sqrt(beta_bar_perp / (6.0 * alpha_perp))
    )


def inside_region(d_perp, d_par, h_par, alpha_perp, beta_bar_perp, epsilon):
    """Return (inside, region_undefined). An undefined boundary counts as inside (no trade)."""
    # Rothert (2024), pp. 68-69
    with np.errstate(divide="ignore", invalid="ignore"):
        h_perp_values = h_perp(d_par, alpha_perp, beta_bar_perp, epsilon)

        region_undefined = (
            ~np.isfinite(h_par)
            | ~np.isfinite(h_perp_values)
            | ~np.isfinite(d_perp)
            | ~np.isfinite(d_par)
        )

        # non-strict: a position projected onto the boundary must count as inside
        raw_inside = (np.abs(d_par) <= h_par) & (np.abs(d_perp) <= h_perp_values)

    inside = np.asarray(np.where(region_undefined, True, raw_inside))
    region_undefined = np.asarray(region_undefined)
    return inside, region_undefined


def project_to_boundary(d_perp, d_par, h_par, alpha_perp, beta_bar_perp, epsilon):
    """Clip a deviation onto the region: parallel coordinate first, then perpendicular."""
    # Rothert (2024), p. 65 (indicator structure); parallel-then-perpendicular is our discretisation
    # parallel first, because h_perp depends on the clipped d_par
    d_par_new = np.clip(d_par, -h_par, h_par)
    h_perp_clipped = h_perp(d_par_new, alpha_perp, beta_bar_perp, epsilon)
    d_perp_new = np.clip(d_perp, -h_perp_clipped, h_perp_clipped)
    return d_perp_new, d_par_new


def compute_rothert_region(paths, config, market=None):
    """Midpoint, coefficients and parallel half width on the whole time grid.

    market: MarketData of paths for the config's strikes and maturities (computed if None).
    """
    _validate_inputs(paths, config)

    if market is None:
        market = prepare_market_data(
            paths, config.K_target, config.K_hedge, config.T_total, config.sigma, config.r,
            hedge_maturity_factor=config.hedge_maturity_factor,
        )

    # diagnostic on the raw gamma, so the floor cannot hide its own flag
    gamma_problematic = np.abs(market.gamma_hedge) < config.gamma_problematic_threshold

    gamma_hedge_effective, gamma_floor_active = _apply_gamma_floor(
        market.gamma_hedge, config.gamma_floor
    )

    phi_stock_star, phi_call_star = midpoint(
        market.delta_target, market.gamma_target, market.delta_hedge, gamma_hedge_effective
    )

    speed_target, speed_hedge = _speed_grid(paths, config)

    # the last time column is degenerate (c_d_par = 0) and tiny hedge gammas can overflow; such
    # NaN/inf values are kept, not repaired, and show up as region_undefined in the strategy
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        c_S_matrix = c_S(paths, config.sigma)
        c_delta_call_matrix = c_delta_call(gamma_hedge_effective, c_S_matrix)
        c_S_perp_matrix = c_S_perp(paths, config.sigma, market.delta_hedge)
        c_d_par_matrix = c_d_par(
            paths, config.sigma, market.gamma_target, gamma_hedge_effective, speed_target, speed_hedge
        )

        alpha_perp_matrix = alpha_perp(
            c_S_perp_matrix, c_delta_call_matrix, market.delta_hedge, config.risk_aversion
        )
        beta_bar_perp_matrix = beta_bar_perp(
            paths, market.call_hedge, market.delta_hedge, alpha_perp_matrix, config.risk_aversion
        )
        alpha_par_matrix = alpha_par(
            beta_bar_perp_matrix, c_delta_call_matrix, c_d_par_matrix, market.delta_hedge,
            paths, market.call_hedge, config.risk_aversion,
        )
        beta_par_matrix = beta_par(
            alpha_par_matrix, paths, market.call_hedge, market.delta_hedge, config.risk_aversion
        )
        h_par_matrix = h_par(alpha_par_matrix, beta_par_matrix, config.epsilon)

    return RothertRegion(
        market=market,
        gamma_hedge=gamma_hedge_effective,
        gamma_problematic=gamma_problematic,
        gamma_floor_active=gamma_floor_active,
        phi_stock_star=phi_stock_star,
        phi_call_star=phi_call_star,
        alpha_perp=alpha_perp_matrix,
        beta_bar_perp=beta_bar_perp_matrix,
        alpha_par=alpha_par_matrix,
        beta_par=beta_par_matrix,
        h_par=h_par_matrix,
    )


def run_rothert_strategy(paths, config, region, cost_fn=None, cost_fn_call=None, call_cost_basis=None):
    """Trade onto the region boundary whenever the held position is outside, otherwise hold.

    cost_fn defaults to proportional costs at rate config.epsilon. cost_fn_call prices the call
    leg (default: cost_fn); call_cost_basis replaces the call price as its cost basis.
    """
    market = region.market

    if cost_fn is None:
        cost_fn = partial(proportional_cost, c=config.epsilon)

    cost_fn_call_used = cost_fn if cost_fn_call is None else cost_fn_call

    if call_cost_basis is not None and tuple(call_cost_basis.shape) != tuple(market.call_hedge.shape):
        raise ValueError(
            f"call_cost_basis.shape {tuple(call_cost_basis.shape)} != "
            f"call_hedge.shape {tuple(market.call_hedge.shape)}"
        )

    n_paths, n_times = paths.shape
    n_steps = n_times - 1

    hedge_gain = np.zeros(n_paths)
    total_costs = np.zeros(n_paths)

    phi_stock_prev = np.zeros(n_paths)
    phi_call_prev = np.zeros(n_paths)

    phi_stock = np.zeros((n_paths, n_steps))
    phi_call = np.zeros((n_paths, n_steps))
    traded = np.zeros((n_paths, n_steps), dtype=bool)
    stock_cap_active = np.zeros((n_paths, n_steps), dtype=bool)
    call_cap_active = np.zeros((n_paths, n_steps), dtype=bool)
    d_par_before_trade = np.zeros((n_paths, n_steps))
    inside_hist = np.zeros((n_paths, n_steps), dtype=bool)
    undefined_hist = np.zeros((n_paths, n_steps), dtype=bool)

    # trading steps only: h_par is NaN in the last time column (c_d_par vanishes at maturity)
    for k in range(n_steps):
        S_k = paths[:, k]
        S_next = paths[:, k + 1]
        C_k = market.call_hedge[:, k]
        C_next = market.call_hedge[:, k + 1]
        dch = market.delta_hedge[:, k]

        d1 = phi_stock_prev - region.phi_stock_star[:, k]
        d2 = phi_call_prev - region.phi_call_star[:, k]

        d_perp_before, d_par_before = to_rothert_coordinates(d1, d2, dch)
        d_par_before_trade[:, k] = d_par_before

        inside_k, undefined_k = inside_region(
            d_perp_before, d_par_before, region.h_par[:, k],
            region.alpha_perp[:, k], region.beta_bar_perp[:, k], config.epsilon,
        )
        inside_hist[:, k] = inside_k
        undefined_hist[:, k] = undefined_k

        d_perp_cand, d_par_cand = project_to_boundary(
            d_perp_before, d_par_before, region.h_par[:, k],
            region.alpha_perp[:, k], region.beta_bar_perp[:, k], config.epsilon,
        )

        d1_new, d2_new = from_rothert_coordinates(d_perp_cand, d_par_cand, dch)
        target_stock = region.phi_stock_star[:, k] + d1_new
        target_call = region.phi_call_star[:, k] + d2_new

        # held positions are carried forward with np.where: a round trip through the
        # coordinates is not bit-exact and would create phantom trades and costs
        phi_stock_k = np.where(inside_k, phi_stock_prev, target_stock)
        phi_call_k = np.where(inside_k, phi_call_prev, target_call)

        # safeguard (not Rothert), applied after the projection; flagged per observation
        phi_stock_k, stock_cap_k = _apply_position_caps(phi_stock_k, config.stock_position_bounds)
        phi_call_k, call_cap_k = _apply_position_caps(phi_call_k, config.call_position_bounds)
        stock_cap_active[:, k] = stock_cap_k
        call_cap_active[:, k] = call_cap_k

        C_k_for_cost = C_k if call_cost_basis is None else call_cost_basis[:, k]
        cost_stock = apply_cost_numpy(cost_fn, phi_stock_k, phi_stock_prev, S_k)
        cost_call = apply_cost_numpy(cost_fn_call_used, phi_call_k, phi_call_prev, C_k_for_cost)
        total_costs += cost_stock
        total_costs += cost_call

        hedge_gain += phi_stock_k * (S_next - S_k) + phi_call_k * (C_next - C_k)

        traded[:, k] = (phi_stock_k != phi_stock_prev) | (phi_call_k != phi_call_prev)

        phi_stock[:, k] = phi_stock_k
        phi_call[:, k] = phi_call_k
        phi_stock_prev, phi_call_prev = phi_stock_k, phi_call_k

    # forced liquidation at maturity: no region decision, no caps
    S_T = paths[:, -1]
    C_T = market.call_hedge[:, -1]
    C_T_for_cost = C_T if call_cost_basis is None else call_cost_basis[:, -1]
    zero = np.zeros(n_paths)
    liquidation_costs = apply_cost_numpy(
        cost_fn, zero, phi_stock_prev, S_T
    ) + apply_cost_numpy(cost_fn_call_used, zero, phi_call_prev, C_T_for_cost)
    total_costs = total_costs + liquidation_costs

    pnl = market.p0 - market.payoff + hedge_gain - total_costs

    return RothertStrategy(
        phi_stock=phi_stock,
        phi_call=phi_call,
        traded=traded,
        inside_region=inside_hist,
        region_undefined=undefined_hist,
        d_par_before_trade=d_par_before_trade,
        stock_cap_active=stock_cap_active,
        call_cap_active=call_cap_active,
        hedge_gain=hedge_gain,
        total_costs=total_costs,
        pnl=pnl,
    )
