import numpy as np
import pytest

from src import config
from src.black_scholes import (
    bs_call_delta, bs_call_gamma, bs_call_price, bs_call_speed, prepare_market_data, simulate_bs_paths,
)

T_HEDGE = config.HEDGE_MATURITY_FACTOR * config.T_TOTAL


def test_reference_values():
    assert bs_call_price(100, 100, config.T_TOTAL, 0.0, 0.2) == pytest.approx(2.2872, abs=5e-5)

    args = (100, 105, T_HEDGE, 0.0, 0.2)
    assert bs_call_delta(*args) == pytest.approx(0.2329910989885145, rel=1e-12)
    assert bs_call_price(*args) == pytest.approx(0.8449919259269834, rel=1e-12)
    assert bs_call_gamma(*args) == pytest.approx(0.04770894873248374, rel=1e-12)

    lambda_star = bs_call_gamma(100.0, 100, config.T_TOTAL, 0.0, 0.2) / bs_call_gamma(100.0, 105, T_HEDGE, 0.0, 0.2)
    assert lambda_star == pytest.approx(1.4578, abs=5e-5)

    # speed is the derivative of gamma
    h = 1e-4
    finite_difference = (
        bs_call_gamma(100.0 + h, 100, config.T_TOTAL, 0.0, 0.2)
        - bs_call_gamma(100.0 - h, 100, config.T_TOTAL, 0.0, 0.2)
    ) / (2 * h)
    assert bs_call_speed(100.0, 100, config.T_TOTAL, 0.0, 0.2) == pytest.approx(finite_difference, rel=1e-6)


def test_paths_and_market_data():
    paths = simulate_bs_paths(n_paths=50, n_steps=10, seed=3)
    assert paths.shape == (50, 11)
    assert np.all(paths[:, 0] == config.S0)
    assert np.array_equal(paths, simulate_bs_paths(n_paths=50, n_steps=10, seed=3))
    assert not np.array_equal(paths, simulate_bs_paths(n_paths=50, n_steps=10, seed=4))

    market = prepare_market_data(
        paths, config.K_TARGET, config.K_HEDGE, config.T_TOTAL, config.SIGMA, config.R,
        hedge_maturity_factor=config.HEDGE_MATURITY_FACTOR,
    )
    for matrix in (market.call_hedge, market.delta_target, market.gamma_target,
                   market.delta_hedge, market.gamma_hedge):
        assert matrix.shape == paths.shape
    assert market.payoff.shape == (50,)
    # the target option has expired in the last column, the hedge call has not
    assert np.all(market.gamma_target[:, -1] == 0.0)
    assert np.all(market.gamma_hedge[:, -1] > 0.0)
