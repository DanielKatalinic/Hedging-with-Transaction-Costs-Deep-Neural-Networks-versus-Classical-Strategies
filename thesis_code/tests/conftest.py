import pytest

from src import config
from src.black_scholes import prepare_market_data, simulate_bs_paths
from src.rothert import RothertConfig


@pytest.fixture(scope="session")
def small_paths():
    return simulate_bs_paths(n_paths=300, seed=7)


@pytest.fixture(scope="session")
def small_market(small_paths):
    return prepare_market_data(
        small_paths, config.K_TARGET, config.K_HEDGE, config.T_TOTAL, config.SIGMA, config.R,
        hedge_maturity_factor=config.HEDGE_MATURITY_FACTOR,
    )


@pytest.fixture
def rothert_config():
    return RothertConfig(
        K_target=config.K_TARGET, K_hedge=config.K_HEDGE, T_total=config.T_TOTAL, sigma=config.SIGMA,
        r=config.R, hedge_maturity_factor=config.HEDGE_MATURITY_FACTOR, epsilon=0.01,
        risk_aversion=config.RISK_AVERSION,
    )
