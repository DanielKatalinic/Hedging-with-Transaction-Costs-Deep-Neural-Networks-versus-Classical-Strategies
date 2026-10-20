from functools import partial

import numpy as np
import pytest
import torch

from src import config
from src.benchmarks import delta_hedge_pnl, gamma_neutral_strategy
from src.cost_functions import apply_cost_numpy, delta_equivalent_call_cost_basis, proportional_cost


def test_cost_functions(small_paths, small_market):
    cost = proportional_cost(torch.tensor([0.5]), torch.tensor([0.2]), torch.tensor([100.0]), c=0.01)
    assert cost.item() == pytest.approx(0.3, rel=1e-6)

    cost_fn = partial(proportional_cost, c=0.01)
    numpy_cost = apply_cost_numpy(cost_fn, np.array([0.5]), np.array([0.2]), np.array([100.0]))
    assert numpy_cost.dtype == np.float32
    assert numpy_cost[0] == pytest.approx(0.3, rel=1e-6)

    # one call on the delta-equivalent basis costs as much as delta_hedge shares of stock
    basis = delta_equivalent_call_cost_basis(small_market.delta_hedge, small_paths)
    one_call = cost_fn(torch.ones(basis.shape), torch.zeros(basis.shape), torch.tensor(basis, dtype=torch.float32))
    delta_shares = cost_fn(
        torch.tensor(small_market.delta_hedge, dtype=torch.float32), torch.zeros(basis.shape),
        torch.tensor(small_paths, dtype=torch.float32),
    )
    np.testing.assert_allclose(one_call.numpy(), delta_shares.numpy(), rtol=1e-6)

    with pytest.raises(ValueError):
        delta_equivalent_call_cost_basis(small_market.delta_hedge, small_paths[:, :-1])


def test_gamma_neutral_and_delta_hedge(small_paths, small_market):
    clip = config.GN_CLIP_LAMBDA
    pnl, stock, call, costs = gamma_neutral_strategy(small_paths, small_market, clip, eps=config.GN_EPS)
    assert np.all(costs == 0.0)

    n_steps = stock.shape[1]
    unclipped = np.abs(call) < clip
    delta_residual = stock + call * small_market.delta_hedge[:, :n_steps] - small_market.delta_target[:, :n_steps]
    gamma_residual = call * small_market.gamma_hedge[:, :n_steps] - small_market.gamma_target[:, :n_steps]
    assert unclipped.mean() > 0.9
    assert np.abs(delta_residual[unclipped]).max() < 1e-12
    np.testing.assert_allclose(gamma_residual[unclipped], 0.0, atol=1e-6)

    # zero cost rate gives the zero-cost P&L, a positive rate only costs money
    zero_cost = partial(proportional_cost, c=0.0)
    pnl_zero, _, _, costs_zero = gamma_neutral_strategy(small_paths, small_market, clip, cost_fn=zero_cost)
    np.testing.assert_allclose(pnl_zero, pnl, rtol=1e-12, atol=1e-12)
    assert np.all(costs_zero == 0.0)
    _, _, _, costs = gamma_neutral_strategy(
        small_paths, small_market, clip, cost_fn=partial(proportional_cost, c=0.01)
    )
    assert np.all(costs >= 0.0) and costs.mean() > 0.0

    pnl_delta, deltas = delta_hedge_pnl(
        small_paths, small_market.payoff, small_market.p0, config.K_TARGET, config.T_TOTAL, config.R, config.SIGMA
    )
    assert np.all(np.isfinite(pnl_delta)) and deltas.shape == stock.shape
