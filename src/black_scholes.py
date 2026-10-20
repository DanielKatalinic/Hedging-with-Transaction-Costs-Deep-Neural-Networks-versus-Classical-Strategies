"""Black-Scholes paths, prices, Greeks and the market data used by all hedging strategies."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


def simulate_bs_paths(S0=100, mu=0.0, sigma=0.2, T=30/365, n_steps=30, n_paths=1000, seed=42):
    """Geometric Brownian motion paths, shape (n_paths, n_steps + 1)."""
    np.random.seed(seed)

    dt = T / n_steps

    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0

    for t in range(n_steps):

        z = np.random.normal(size=n_paths)

        paths[:, t + 1] = (paths[:, t] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        )

    return paths


def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price; intrinsic value for T <= 0."""
    if T <= 0:
        return np.maximum(S - K, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    call = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))

    return call


def call_payoff(S_T, K):
    return np.maximum(S_T - K, 0)


def bs_call_delta(S, K, T, r, sigma):
    """Black-Scholes call delta; step function for T <= 0."""
    if T <= 0:
        return np.where(S > K, 1.0, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    return norm.cdf(d1)


def bs_call_gamma(S, K, T, r, sigma):
    """Black-Scholes call gamma, vectorised over S; zero for T <= 0."""
    S = np.asarray(S, dtype=float)
    if float(T) <= 0:
        return np.zeros_like(S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_call_speed(S, K, T, r, sigma):
    """Speed = dGamma/dS = -Gamma/S * (1 + d1/(sigma*sqrt(T))); standard Black-Scholes Greek, not from Rothert."""
    S = np.asarray(S, dtype=float)
    if float(T) <= 0:
        return np.zeros_like(S)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    # gamma recomputed locally so the finite-difference test of speed stays independent
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    speed = -gamma / S * (1.0 + d1 / (sigma * np.sqrt(T)))
    return speed


@dataclass
class MarketData:
    """Payoff, initial premium and Greeks of target and hedge call; matrices are (n_paths, n_times)."""
    payoff: np.ndarray
    p0: float
    call_hedge: np.ndarray
    delta_target: np.ndarray
    gamma_target: np.ndarray
    delta_hedge: np.ndarray
    gamma_hedge: np.ndarray


def prepare_market_data(paths, K_target, K_hedge, T_total, sigma, r, hedge_maturity_factor=1.0, S0=100):
    """Price and Greeks of the target call (maturity T_total) and the hedge call
    (maturity hedge_maturity_factor * T_total) along every path."""
    n_paths, n_times = paths.shape
    n_steps = n_times - 1
    T_hedge = T_total * hedge_maturity_factor

    payoff = call_payoff(paths[:, -1], K_target)
    p0 = float(bs_call_price(S=S0, K=K_target, T=T_total, sigma=sigma, r=r))

    call_hedge = np.zeros_like(paths)
    delta_target = np.zeros_like(paths)
    gamma_target = np.zeros_like(paths)
    delta_hedge = np.zeros_like(paths)
    gamma_hedge = np.zeros_like(paths)

    for k in range(n_times):
        t_k = k * T_total / n_steps
        tau = T_total - t_k
        tau_hedge = T_hedge - t_k
        S_k = paths[:, k]

        call_hedge[:, k] = bs_call_price(S_k, K_hedge, tau_hedge, r, sigma)
        delta_target[:, k] = bs_call_delta(S_k, K_target, tau, r, sigma)
        delta_hedge[:, k] = bs_call_delta(S_k, K_hedge, tau_hedge, r, sigma)
        gamma_target[:, k] = bs_call_gamma(S_k, K_target, tau, r, sigma)
        gamma_hedge[:, k] = bs_call_gamma(S_k, K_hedge, tau_hedge, r, sigma)

    return MarketData(payoff, p0, call_hedge, delta_target, gamma_target, delta_hedge, gamma_hedge)
