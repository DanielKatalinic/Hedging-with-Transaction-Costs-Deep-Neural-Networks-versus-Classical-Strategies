"""Hedging networks, their P&L functions, losses, and saving and loading of checkpoints."""

import torch
import torch.nn as nn


class HedgeNet(nn.Module):
    """Feed-forward network for one asset. Input (log(S/K), tau/T), output delta."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


class RecurrentHedgeNet(nn.Module):
    """One asset with the previous delta as extra input (needed once trading costs matter)."""

    def __init__(self, market_dim, hidden_dim):
        super().__init__()

        input_dim = market_dim + 1

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, 1)
        )

    def forward(self, market_features, prev_delta):

        x = torch.cat(
            [market_features, prev_delta],
            dim=1
        )

        return self.net(x)


class TwoAssetHedgeNet(nn.Module):
    """Stock and hedge call without memory. Output (delta_stock, delta_call)."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return self.net(x)


class RecurrentTwoAssetHedgeNet(nn.Module):
    """Stock and hedge call with both previous positions as extra inputs."""

    def __init__(self, market_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(market_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, market_features, prev_delta_stock, prev_delta_call):
        x = torch.cat([market_features, prev_delta_stock, prev_delta_call], dim=1)
        return self.net(x)


def pnl_single_asset(model, paths, payoff, p0, T_total, K):
    """Terminal P&L of a one-asset network without costs, shape (n_paths,)."""
    n_paths, n_times = paths.shape
    n_steps = n_times - 1

    hedge_gain = torch.zeros(n_paths, dtype=torch.float32)

    for k in range(n_steps):
        S_k = paths[:, k]
        S_next = paths[:, k + 1]

        tau = T_total - k * T_total / n_steps
        tau_tensor = torch.full_like(S_k, tau)

        x = torch.stack(
            [torch.log(S_k/K), tau_tensor/T_total],
            dim=1
        )

        delta_k = model(x).squeeze()

        hedge_gain = hedge_gain + delta_k * (S_next - S_k)

    pnl = p0 - payoff + hedge_gain

    return pnl


def pnl_single_asset_with_costs(model, paths, payoff, p0, T_total, K, cost_fn):
    """Terminal P&L of a recurrent one-asset network with costs and final liquidation.

    cost_fn(delta_new, delta_prev, price) -> (n_paths,). Returns (pnl, delta_history, total_costs).
    """
    n_paths, n_times = paths.shape
    n_steps = n_times - 1

    hedge_gain    = torch.zeros(n_paths, dtype=torch.float32, device=paths.device)
    total_costs   = torch.zeros(n_paths, dtype=torch.float32, device=paths.device)
    delta_prev    = torch.zeros(n_paths, 1, dtype=torch.float32, device=paths.device)
    delta_history = torch.zeros(n_paths, n_steps, device=paths.device)

    for k in range(n_steps):
        S_k    = paths[:, k]
        S_next = paths[:, k + 1]

        tau        = T_total - k * T_total / n_steps
        tau_tensor = torch.full_like(S_k, tau)

        market_features = torch.stack(
            [torch.log(S_k / K), tau_tensor / T_total],
            dim=1
        )

        delta_k = model(market_features, delta_prev)

        total_costs          += cost_fn(delta_k.squeeze(), delta_prev.squeeze(), S_k)
        hedge_gain           += delta_k.squeeze() * (S_next - S_k)
        delta_history[:, k]   = delta_k.squeeze().detach()

        delta_prev = delta_k

    S_T = paths[:, -1]
    total_costs += cost_fn(
        torch.zeros(n_paths, device=paths.device),
        delta_prev.squeeze(),
        S_T
    )

    pnl = p0 - payoff + hedge_gain - total_costs

    return pnl, delta_history, total_costs


def pnl_two_assets(model, paths, call_prices, payoff, p0, T_total, K_target):
    """Terminal P&L of a two-asset network without costs. Returns (pnl, stock_history, call_history)."""
    n_paths, n_times = paths.shape
    n_steps = n_times - 1

    hedge_gain = torch.zeros(n_paths, dtype=torch.float32, device=paths.device)

    delta_stock_history = torch.zeros(
        n_paths,
        n_steps,
        device=paths.device
    )

    delta_call_history = torch.zeros(
        n_paths,
        n_steps,
        device=paths.device
    )

    for k in range(n_steps):
        S_k = paths[:, k]
        S_next = paths[:, k + 1]

        C_k = call_prices[:, k]
        C_next = call_prices[:, k + 1]

        tau = T_total - k * T_total / n_steps
        tau_tensor = torch.full_like(S_k, tau)

        x = torch.stack([
            torch.log(S_k / K_target),
            C_k / S_k,
            tau_tensor / T_total
        ], dim=1)

        delta = model(x)

        delta_stock = delta[:, 0]
        delta_call = delta[:, 1]

        delta_stock_history[:,k] = delta_stock.detach()
        delta_call_history[:,k] = delta_call.detach()

        hedge_gain += (delta_stock*(S_next-S_k)+ delta_call*(C_next-C_k))

    pnl = p0 - payoff + hedge_gain

    return pnl, delta_stock_history, delta_call_history


def pnl_two_assets_with_costs(model, paths, call_prices, payoff, p0, T_total, K_target, cost_fn,
                              cost_fn_call=None, call_cost_basis=None):
    """Terminal P&L of a recurrent two-asset network with costs on both legs and final liquidation.

    cost_fn prices the stock leg. cost_fn_call prices the call leg (default: cost_fn).
    call_cost_basis (n_paths, n_times) replaces the call price as the call leg's cost basis;
    gains always use the call price. Returns (pnl, stock_history, call_history, total_costs).
    """
    cost_fn_call_used = cost_fn if cost_fn_call is None else cost_fn_call

    if call_cost_basis is not None and tuple(call_cost_basis.shape) != tuple(call_prices.shape):
        raise ValueError(
            f"call_cost_basis.shape {tuple(call_cost_basis.shape)} != "
            f"call_prices.shape {tuple(call_prices.shape)}"
        )

    n_paths, n_times = paths.shape
    n_steps = n_times - 1

    hedge_gain          = torch.zeros(n_paths, dtype=torch.float32, device=paths.device)
    total_costs         = torch.zeros(n_paths, dtype=torch.float32, device=paths.device)
    delta_stock_prev    = torch.zeros(n_paths, 1, device=paths.device)
    delta_call_prev     = torch.zeros(n_paths, 1, device=paths.device)
    delta_stock_history = torch.zeros(n_paths, n_steps, device=paths.device)
    delta_call_history  = torch.zeros(n_paths, n_steps, device=paths.device)

    for k in range(n_steps):
        S_k    = paths[:, k]
        S_next = paths[:, k + 1]
        C_k    = call_prices[:, k]
        C_next = call_prices[:, k + 1]

        tau        = T_total - k * T_total / n_steps
        tau_tensor = torch.full_like(S_k, tau)

        market_features = torch.stack([
            torch.log(S_k / K_target),
            C_k / S_k,
            tau_tensor / T_total
        ], dim=1)

        delta = model(market_features, delta_stock_prev, delta_call_prev)
        delta_stock = delta[:, 0]
        delta_call  = delta[:, 1]

        call_price_for_cost = C_k if call_cost_basis is None else call_cost_basis[:, k]
        total_costs += cost_fn(delta_stock,          delta_stock_prev.squeeze(), S_k)
        total_costs += cost_fn_call_used(delta_call, delta_call_prev.squeeze(), call_price_for_cost)
        hedge_gain  += delta_stock * (S_next - S_k) + delta_call * (C_next - C_k)

        delta_stock_history[:, k] = delta_stock.detach()
        delta_call_history[:, k]  = delta_call.detach()

        delta_stock_prev = delta_stock.unsqueeze(1)
        delta_call_prev  = delta_call.unsqueeze(1)

    S_T = paths[:, -1]
    C_T = call_prices[:, -1]
    C_T_for_cost = C_T if call_cost_basis is None else call_cost_basis[:, -1]
    total_costs += cost_fn(torch.zeros(n_paths, device=paths.device), delta_stock_prev.squeeze(), S_T)
    total_costs += cost_fn_call_used(
        torch.zeros(n_paths, device=paths.device), delta_call_prev.squeeze(), C_T_for_cost
    )

    pnl = p0 - payoff + hedge_gain - total_costs

    return pnl, delta_stock_history, delta_call_history, total_costs


def entropic_loss(x, lam):
    """Entropic risk (1/lam) log E[exp(-lam x)], computed with logsumexp."""
    return (1/lam) * (
        torch.logsumexp(-lam * x, dim=0)
        - torch.log(torch.tensor(x.numel(), dtype=x.dtype, device=x.device))
    )


def mse_loss(pnl):
    return (pnl ** 2).mean()


def save_model(model, path, fingerprint, loss_history):
    """Write the weights, the fingerprint (dict of the training settings) and the loss history to path.

    scripts/train.py saves every network with this function; load_model reads the file back.
    """
    torch.save(
        {"state_dict": model.state_dict(), "fingerprint": fingerprint, "loss_history": loss_history},
        path,
    )
    print(f"saved {path}")


def load_model(path):
    """Load a network from a checkpoint file; the class follows from the stored layer sizes.

    The stored fingerprint is not checked here (scripts/figures.py and scripts/tables.py use this);
    tests/test_deep_hedging.py compares the fingerprints of the shipped files with the ones scripts/train.py
    writes. A file that does not fit the chosen class fails in load_state_dict.
    """
    # weights_only: the file holds only tensors and primitive metadata
    state_dict = torch.load(path, weights_only=True)["state_dict"]
    hidden_dim, n_inputs = state_dict["net.0.weight"].shape  # first layer: (hidden, inputs)
    n_outputs = state_dict["net.4.weight"].shape[0]  # last layer: (outputs, hidden)

    if (n_inputs, n_outputs) == (2, 1):
        model = HedgeNet(input_dim=2, hidden_dim=hidden_dim)  # log(S/K), tau/T
    elif (n_inputs, n_outputs) == (3, 1):
        model = RecurrentHedgeNet(market_dim=2, hidden_dim=hidden_dim)  # log(S/K), tau/T, previous delta
    elif (n_inputs, n_outputs) == (3, 2):
        model = TwoAssetHedgeNet(input_dim=3, hidden_dim=hidden_dim)  # log(S/K), C/S, tau/T
    elif (n_inputs, n_outputs) == (5, 2):
        model = RecurrentTwoAssetHedgeNet(market_dim=3, hidden_dim=hidden_dim)  # plus both previous positions
    else:
        raise ValueError(f"{path}: no network class for {n_inputs} inputs and {n_outputs} outputs")

    model.load_state_dict(state_dict)
    model.eval()
    return model
