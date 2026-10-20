"""Proportional transaction costs and the delta-equivalent cost basis of the call leg."""

import torch


def proportional_cost(delta_new, delta_prev, price, c):
    """Cost c * price * |delta_new - delta_prev| (torch tensors)."""
    dn = delta_new - delta_prev
    return c * price * torch.abs(dn)


def delta_equivalent_call_cost_basis(delta_hedge, S):
    """Cost basis delta_hedge * S for the call leg, so one unit of delta costs the same
    through the call as through the stock. Shapes must match (no broadcasting)."""
    if tuple(delta_hedge.shape) != tuple(S.shape):
        raise ValueError(
            f"delta_hedge.shape {tuple(delta_hedge.shape)} != S.shape {tuple(S.shape)}: "
            "the cost basis expects equal shapes, no broadcasting."
        )
    return delta_hedge * S


def apply_cost_numpy(cost_fn, new, prev, price):
    """Apply a torch cost function to numpy arrays and return numpy.

    The numpy strategies pay costs through the same float32 torch function as the network.
    """
    result = cost_fn(
        torch.tensor(new, dtype=torch.float32),
        torch.tensor(prev, dtype=torch.float32),
        torch.tensor(price, dtype=torch.float32),
    )
    return result.detach().numpy()
