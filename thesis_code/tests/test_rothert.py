import dataclasses

import numpy as np
import pytest

from src.rothert import (
    c_S_perp, compute_rothert_region, from_rothert_coordinates, h_par, h_perp, inside_region,
    midpoint, run_rothert_strategy, to_rothert_coordinates,
)


def test_hand_computed_anchors():
    # sigma^2 S^2 (1 + C'^2)^2 = 400 * 1.5625; the squared factor (1 + C'^2)^2, not 1.35^2 times 400
    assert c_S_perp(100.0, 0.2, 0.5) == pytest.approx(625.0, rel=1e-12)
    assert c_S_perp(100.0, 0.2, 0.5) != pytest.approx(729.0)

    # eps^(2/7) |d_par|^(2/3) sqrt(beta / (6 alpha)); the exponent 2/7 is the printed one, 1/3 is not
    assert h_perp(0.5, -3.0, -60.0, 0.01) == pytest.approx(0.3085489958942867, rel=1e-12)
    assert h_perp(0.5, -3.0, -60.0, 0.01) != pytest.approx(0.24779129318307333)

    # eps^(1/7) sqrt(beta / (6 alpha)); 9/49 would give another value
    assert h_par(-2.0, -50.0, 0.01) == pytest.approx(1.0572558416485038, rel=1e-12)
    assert h_par(-2.0, -50.0, 0.01) != pytest.approx(0.8760874122404292)


def test_region_geometry(small_paths, small_market, rothert_config):
    region = compute_rothert_region(small_paths, rothert_config, small_market)

    # the midpoint is delta- and gamma-neutral
    market = small_market
    stock, call = midpoint(market.delta_target, market.gamma_target, market.delta_hedge, market.gamma_hedge)
    np.testing.assert_allclose(stock + call * market.delta_hedge, market.delta_target, atol=1e-12)
    np.testing.assert_allclose(call * market.gamma_hedge, market.gamma_target, atol=1e-12)
    assert np.array_equal(stock, region.phi_stock_star) and np.array_equal(call, region.phi_call_star)

    # coordinate round trip
    rng = np.random.default_rng(0)
    d1, d2, dch = rng.normal(size=50), rng.normal(size=50), rng.uniform(0.1, 0.9, size=50)
    back = from_rothert_coordinates(*to_rothert_coordinates(d1, d2, dch), dch)
    np.testing.assert_allclose(back, (d1, d2), atol=1e-12)

    k = 5
    alpha_perp, beta_bar_perp = region.alpha_perp[:, k], region.beta_bar_perp[:, k]
    h = region.h_par[:, k]
    eps = rothert_config.epsilon
    zero = np.zeros_like(h)

    inside, undefined = inside_region(zero, zero, h, alpha_perp, beta_bar_perp, eps)
    assert inside.all() and not undefined.any()

    # symmetry under a sign flip of both coordinates
    d_perp, d_par = rng.normal(size=h.size) * 0.1, rng.normal(size=h.size) * h
    a = inside_region(d_perp, d_par, h, alpha_perp, beta_bar_perp, eps)[0]
    b = inside_region(-d_perp, -d_par, h, alpha_perp, beta_bar_perp, eps)[0]
    assert np.array_equal(a, b)

    # at the waist d_par = 0 the region has no width: any perpendicular deviation is outside
    waist, waist_undefined = inside_region(np.full_like(h, 1e-6), zero, h, alpha_perp, beta_bar_perp, eps)
    assert (~waist_undefined).any()
    assert not waist[~waist_undefined].any()

    # an undefined boundary counts as inside and is reported
    nan_inside, nan_undefined = inside_region(
        np.array([1.0]), np.array([1.0]), np.array([np.nan]), np.array([-1.0]), np.array([-1.0]), eps
    )
    assert nan_inside.all() and nan_undefined.all()


def test_strategy_trades_only_outside(small_paths, small_market, rothert_config):
    region = compute_rothert_region(small_paths, rothert_config, small_market)
    strategy = run_rothert_strategy(small_paths, rothert_config, region)
    n_steps = small_paths.shape[1] - 1
    eps = rothert_config.epsilon

    assert strategy.traded.any() and (~strategy.traded).any()

    # no trade where the held position is inside: bit-identical carry-forward
    stock_prev = np.concatenate([np.zeros((small_paths.shape[0], 1)), strategy.phi_stock[:, :-1]], axis=1)
    call_prev = np.concatenate([np.zeros((small_paths.shape[0], 1)), strategy.phi_call[:, :-1]], axis=1)
    held = strategy.inside_region
    assert np.array_equal(strategy.phi_stock[held], stock_prev[held])
    assert np.array_equal(strategy.phi_call[held], call_prev[held])

    # every traded position lies inside the region at the same step
    d1 = strategy.phi_stock - region.phi_stock_star[:, :n_steps]
    d2 = strategy.phi_call - region.phi_call_star[:, :n_steps]
    d_perp, d_par = to_rothert_coordinates(d1, d2, small_market.delta_hedge[:, :n_steps])
    h_perp_values = h_perp(d_par, region.alpha_perp[:, :n_steps], region.beta_bar_perp[:, :n_steps], eps)
    checked = strategy.traded & ~strategy.region_undefined
    assert checked.any()
    assert (np.abs(d_par) <= region.h_par[:, :n_steps] + 1e-9)[checked].all()
    assert (np.abs(d_perp) <= h_perp_values + 1e-9)[checked].all()

    np.testing.assert_allclose(
        strategy.pnl, small_market.p0 - small_market.payoff + strategy.hedge_gain - strategy.total_costs,
        rtol=1e-12,
    )

    # safeguards are off by default and flag nothing
    assert not (strategy.stock_cap_active.any() or strategy.call_cap_active.any())
    assert not region.gamma_floor_active.any()

    # position caps clip the held position and flag exactly the clipped observations
    capped_config = dataclasses.replace(rothert_config, stock_position_bounds=(-0.5, 0.5))
    capped = run_rothert_strategy(small_paths, capped_config, region)
    assert capped.stock_cap_active.any()
    assert np.abs(capped.phi_stock).max() <= 0.5
    assert np.all(np.abs(capped.phi_stock[capped.stock_cap_active]) == 0.5)
