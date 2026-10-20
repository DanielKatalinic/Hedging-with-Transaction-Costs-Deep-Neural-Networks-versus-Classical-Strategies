"""Retrains the ten networks behind the thesis figures and tables and saves them into OUT_DIR.

Run from the thesis_code folder: python -m scripts.train (py -m scripts.train on Windows). All ten networks take about 55 min on
the CPU; set N_EPOCHS_OVERRIDE = 50 for a quick test of about 35 s. The shipped networks in models/
are never replaced: OUT_DIR is a separate folder and the script stops if it points to models/. Each
network has its own section below in thesis order; remove its name from TRAIN to skip it. To use the
retrained networks, set MODELS_DIR = "models_retrained" at the top of scripts/figures.py or scripts/tables.py.
"""

from functools import partial
from pathlib import Path

import torch

from src.black_scholes import bs_call_price, call_payoff, prepare_market_data, simulate_bs_paths
from src.config import (
    HEDGE_MATURITY_FACTOR, K_HEDGE, K_SINGLE, K_TARGET, MU, N_STEPS, N_STEPS_FINE, N_TRAIN, R,
    RISK_AVERSION, S0, SEED_TRAIN_PATHS, SIGMA, T_TOTAL,
)
from src.cost_functions import delta_equivalent_call_cost_basis, proportional_cost
from src.deep_hedging import (
    HedgeNet, RecurrentHedgeNet, RecurrentTwoAssetHedgeNet, TwoAssetHedgeNet, entropic_loss, mse_loss,
    pnl_single_asset, pnl_single_asset_with_costs, pnl_two_assets, pnl_two_assets_with_costs, save_model,
)

OUT_DIR = "models_retrained"  # a separate folder: models/ holds the shipped networks
TRAIN = [  # remove a name to skip that network
    "single_asset_no_costs", "two_assets_no_costs", "single_asset_costs", "kappa_sweep", "fine_grid_120",
    "delta_neutral_basis",
]
N_EPOCHS_OVERRIDE = None  # e.g. 50 for a quick test; None trains every network for its full number of epochs

if Path(OUT_DIR).resolve() == Path("models").resolve():
    raise SystemExit("OUT_DIR must not be models/: that folder holds the shipped networks")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# 1) training paths (seed 111), shared by all networks with 30 trading steps
paths = simulate_bs_paths(
    S0=S0, mu=MU, sigma=SIGMA, T=T_TOTAL, n_steps=N_STEPS, n_paths=N_TRAIN, seed=SEED_TRAIN_PATHS
)
paths_t = torch.tensor(paths, dtype=torch.float32)

# 2) one asset: the sold call with strike K_SINGLE, hedged with the stock only
payoff_single_t = torch.tensor(call_payoff(paths[:, -1], K_SINGLE), dtype=torch.float32)
p0_single_t = torch.tensor(bs_call_price(S=S0, K=K_SINGLE, T=T_TOTAL, r=R, sigma=SIGMA), dtype=torch.float32)

# 3) two assets: the sold call K_TARGET, hedged with the stock and the hedge call K_HEDGE
market = prepare_market_data(
    paths, K_TARGET, K_HEDGE, T_TOTAL, SIGMA, R, hedge_maturity_factor=HEDGE_MATURITY_FACTOR
)
payoff_t = torch.tensor(market.payoff, dtype=torch.float32)
p0_t = torch.tensor(market.p0, dtype=torch.float32)
call_t = torch.tensor(market.call_hedge, dtype=torch.float32)


# --- Network: single_asset_no_costs, stock only, no costs (used in Figure 1 / Table 1) ---
if "single_asset_no_costs" in TRAIN:
    n_epochs = 15000 if N_EPOCHS_OVERRIDE is None else N_EPOCHS_OVERRIDE
    batch_size, lr, lam = 256, 0.0005, 1.0
    seed_weights, seed_batches = 20260919, 20260920
    print(f"single_asset_no_costs: {n_epochs} epochs")
    torch.manual_seed(seed_weights)  # initial weights
    model = HedgeNet(input_dim=2, hidden_dim=32)  # inputs log(S/K) and tau/T
    generator = torch.Generator().manual_seed(seed_batches)  # order of the mini-batches
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(n_epochs):
        idx = torch.randint(0, N_TRAIN, (batch_size,), generator=generator)
        optimizer.zero_grad()
        pnl = pnl_single_asset(
            model, paths=paths_t[idx], payoff=payoff_single_t[idx], p0=p0_single_t, T_total=T_TOTAL, K=K_SINGLE,
        )
        loss = entropic_loss(pnl, lam=lam)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 500 == 0:
            print(epoch, round(loss.item(), 5))
    fingerprint = {  # the settings of this run, stored in the checkpoint (key names as in the original run)
        "input_dim": 2, "hidden_dim": 32, "pnl_fn": "compute_nn_pnl_no_costs", "loss_name": "entropic_loss",
        "risk_aversion": lam, "n_epochs": n_epochs, "batch_size": batch_size, "lr": lr,
        "seed_weights": seed_weights, "seed_batches": seed_batches, "n_train": N_TRAIN,
        "seed_train_paths": SEED_TRAIN_PATHS, "K_feature": K_SINGLE, "K_payoff": K_SINGLE,
        "T_total": T_TOTAL, "sigma": SIGMA, "r": R,
    }
    save_model(model, f"{OUT_DIR}/hedgenet_nb01_entropic_l1.pt", fingerprint, losses)


# --- Network: two_assets_no_costs, stock and hedge call, no costs (used in Figure 2 / Table 1) ---
# The shipped network was trained without seeds, so a retrain gives other weights.
if "two_assets_no_costs" in TRAIN:
    n_epochs = 15000 if N_EPOCHS_OVERRIDE is None else N_EPOCHS_OVERRIDE
    batch_size, lr = 256, 0.001
    print(f"two_assets_no_costs: {n_epochs} epochs")
    model = TwoAssetHedgeNet(input_dim=3, hidden_dim=64)  # inputs log(S/K), C/S and tau/T
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(n_epochs):
        idx = torch.randint(0, N_TRAIN, (batch_size,))  # no own generator: global torch random numbers
        optimizer.zero_grad()
        pnl, _, _ = pnl_two_assets(
            model, paths=paths_t[idx], call_prices=call_t[idx], payoff=payoff_t[idx], p0=p0_t,
            T_total=T_TOTAL, K_target=K_TARGET,
        )
        loss = mse_loss(pnl)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 500 == 0:
            print(epoch, round(loss.item(), 5))
    fingerprint = {
        "input_dim": 3, "hidden_dim": 64, "pnl_fn": "compute_nn_pnl_two_assets", "loss_name": "mse_loss",
        "n_epochs": n_epochs, "batch_size": batch_size, "lr": lr, "seed_weights": None, "seed_batches": None,
        "n_train": N_TRAIN, "seed_train_paths": SEED_TRAIN_PATHS, "K_target": K_TARGET, "K_hedge": K_HEDGE,
        "T_total": T_TOTAL, "sigma": SIGMA, "r": R, "hedge_maturity_factor": HEDGE_MATURITY_FACTOR,
    }
    save_model(model, f"{OUT_DIR}/hedgenet_nb07_mse_3feat.pt", fingerprint, losses)


# --- Network: single_asset_costs, stock only, costs kappa = 0.01 (used in Figure 3) ---
if "single_asset_costs" in TRAIN:
    n_epochs = 5000 if N_EPOCHS_OVERRIDE is None else N_EPOCHS_OVERRIDE
    batch_size, lr, lam, kappa = 256, 0.001, 1.0, 0.01
    seed_weights, seed_batches = 20260922, 20260923
    print(f"single_asset_costs: {n_epochs} epochs")
    cost_fn = partial(proportional_cost, c=kappa)
    torch.manual_seed(seed_weights)
    model = RecurrentHedgeNet(market_dim=2, hidden_dim=32)  # inputs log(S/K), tau/T and the previous delta
    generator = torch.Generator().manual_seed(seed_batches)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(n_epochs):
        idx = torch.randint(0, N_TRAIN, (batch_size,), generator=generator)
        optimizer.zero_grad()
        pnl, _, _ = pnl_single_asset_with_costs(
            model, paths=paths_t[idx], payoff=payoff_single_t[idx], p0=p0_single_t, T_total=T_TOTAL, K=K_SINGLE,
            cost_fn=cost_fn,
        )
        loss = entropic_loss(pnl, lam=lam)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 500 == 0:
            print(epoch, round(loss.item(), 5))
    fingerprint = {
        "market_dim": 2, "hidden_dim": 32, "pnl_fn": "compute_nn_pnl_with_costs", "loss_name": "entropic_loss",
        "lam": lam, "n_epochs": n_epochs, "batch_size": batch_size, "lr": lr, "seed_weights": seed_weights,
        "seed_batches": seed_batches, "n_train": N_TRAIN, "seed_train_paths": SEED_TRAIN_PATHS, "K": K_SINGLE,
        "T_total": T_TOTAL, "sigma": SIGMA, "r": R, "kappa": kappa,
    }
    save_model(model, f"{OUT_DIR}/hedgenet_nb05_entropic_kappa1e-02.pt", fingerprint, losses)


# --- Network: kappa_sweep, one network per kappa (used in Figures 4-10 / Tables 2-4) ---
# Stock and hedge call with costs on both legs. The kappa = 0.01 network is the baseline network.
if "kappa_sweep" in TRAIN:
    n_epochs = 10000 if N_EPOCHS_OVERRIDE is None else N_EPOCHS_OVERRIDE
    batch_size, lr = 256, 0.001
    seed_weights, seed_batches = 20260810, 20260811  # the same seeds for every kappa
    for kappa in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]:
        print(f"kappa_sweep, kappa {kappa:g}: {n_epochs} epochs")
        cost_fn = partial(proportional_cost, c=kappa)
        torch.manual_seed(seed_weights)
        model = RecurrentTwoAssetHedgeNet(market_dim=3, hidden_dim=64)  # plus both previous positions
        generator = torch.Generator().manual_seed(seed_batches)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        losses = []
        for epoch in range(n_epochs):
            idx = torch.randint(0, N_TRAIN, (batch_size,), generator=generator)
            optimizer.zero_grad()
            pnl, _, _, _ = pnl_two_assets_with_costs(
                model, paths=paths_t[idx], call_prices=call_t[idx], payoff=payoff_t[idx], p0=p0_t,
                T_total=T_TOTAL, K_target=K_TARGET, cost_fn=cost_fn,
            )
            loss = entropic_loss(pnl, lam=RISK_AVERSION)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if epoch % 500 == 0:
                print(epoch, round(loss.item(), 5))
        fingerprint = {
            "market_dim": 3, "hidden_dim": 64, "loss_name": "entropic_loss", "risk_aversion": RISK_AVERSION,
            "n_epochs": n_epochs, "batch_size": batch_size, "lr": lr, "seed_weights": seed_weights,
            "seed_batches": seed_batches, "K_target": K_TARGET, "K_hedge": K_HEDGE, "T_total": T_TOTAL,
            "sigma": SIGMA, "r": R, "kappa": kappa, "hedge_maturity_factor": HEDGE_MATURITY_FACTOR,
        }
        save_model(model, f"{OUT_DIR}/hedgenet_kappa_sweep_{kappa:.0e}.pt", fingerprint, losses)


# --- Network: fine_grid_120, the kappa = 0.01 network on 120 trading steps (used in Figure 9 / Table 4) ---
if "fine_grid_120" in TRAIN:
    paths_fine = simulate_bs_paths(
        S0=S0, mu=MU, sigma=SIGMA, T=T_TOTAL, n_steps=N_STEPS_FINE, n_paths=N_TRAIN, seed=SEED_TRAIN_PATHS
    )
    market_fine = prepare_market_data(
        paths_fine, K_TARGET, K_HEDGE, T_TOTAL, SIGMA, R, hedge_maturity_factor=HEDGE_MATURITY_FACTOR
    )
    paths_fine_t = torch.tensor(paths_fine, dtype=torch.float32)
    payoff_fine_t = torch.tensor(market_fine.payoff, dtype=torch.float32)
    p0_fine_t = torch.tensor(market_fine.p0, dtype=torch.float32)
    call_fine_t = torch.tensor(market_fine.call_hedge, dtype=torch.float32)
    n_epochs = 10000 if N_EPOCHS_OVERRIDE is None else N_EPOCHS_OVERRIDE
    batch_size, lr, kappa = 256, 0.001, 0.01
    seed_weights, seed_batches = 20260810, 20260811
    print(f"fine_grid_120: {n_epochs} epochs")
    cost_fn = partial(proportional_cost, c=kappa)
    torch.manual_seed(seed_weights)
    model = RecurrentTwoAssetHedgeNet(market_dim=3, hidden_dim=64)
    generator = torch.Generator().manual_seed(seed_batches)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(n_epochs):
        idx = torch.randint(0, N_TRAIN, (batch_size,), generator=generator)
        optimizer.zero_grad()
        pnl, _, _, _ = pnl_two_assets_with_costs(
            model, paths=paths_fine_t[idx], call_prices=call_fine_t[idx], payoff=payoff_fine_t[idx],
            p0=p0_fine_t, T_total=T_TOTAL, K_target=K_TARGET, cost_fn=cost_fn,
        )
        loss = entropic_loss(pnl, lam=RISK_AVERSION)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 500 == 0:
            print(epoch, round(loss.item(), 5))
    fingerprint = {
        "market_dim": 3, "hidden_dim": 64, "loss_name": "entropic_loss", "risk_aversion": RISK_AVERSION,
        "n_epochs": n_epochs, "batch_size": batch_size, "lr": lr, "seed_weights": seed_weights,
        "seed_batches": seed_batches, "n_steps": N_STEPS_FINE, "K_target": K_TARGET, "K_hedge": K_HEDGE,
        "T_total": T_TOTAL, "sigma": SIGMA, "r": R, "kappa": kappa, "hedge_maturity_factor": HEDGE_MATURITY_FACTOR,
    }
    save_model(model, f"{OUT_DIR}/hedgenet_nb09_n120_kappa0.01.pt", fingerprint, losses)


# --- Network: delta_neutral_basis, other cost basis of the call leg (used in Figure 10 / Table 4) ---
# As the kappa = 0.01 network, but the call leg pays costs on delta_hedge * S instead of the call price.
# The gains still use the call price.
if "delta_neutral_basis" in TRAIN:
    basis_t = torch.tensor(delta_equivalent_call_cost_basis(market.delta_hedge, paths), dtype=torch.float32)
    n_epochs = 10000 if N_EPOCHS_OVERRIDE is None else N_EPOCHS_OVERRIDE
    batch_size, lr, kappa = 256, 0.001, 0.01
    seed_weights, seed_batches = 20260810, 20260811
    print(f"delta_neutral_basis: {n_epochs} epochs")
    cost_fn = partial(proportional_cost, c=kappa)
    torch.manual_seed(seed_weights)
    model = RecurrentTwoAssetHedgeNet(market_dim=3, hidden_dim=64)
    generator = torch.Generator().manual_seed(seed_batches)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(n_epochs):
        idx = torch.randint(0, N_TRAIN, (batch_size,), generator=generator)
        optimizer.zero_grad()
        pnl, _, _, _ = pnl_two_assets_with_costs(
            model, paths=paths_t[idx], call_prices=call_t[idx], payoff=payoff_t[idx], p0=p0_t,
            T_total=T_TOTAL, K_target=K_TARGET, cost_fn=cost_fn, call_cost_basis=basis_t[idx],
        )
        loss = entropic_loss(pnl, lam=RISK_AVERSION)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 500 == 0:
            print(epoch, round(loss.item(), 5))
    fingerprint = {
        "market_dim": 3, "hidden_dim": 64, "loss_name": "entropic_loss", "risk_aversion": RISK_AVERSION,
        "n_epochs": n_epochs, "batch_size": batch_size, "lr": lr, "seed_weights": seed_weights,
        "seed_batches": seed_batches, "n_steps": N_STEPS,
        "cost_model": "stock_proportional__call_delta_equivalent_basis", "fee": 0.0,
        "K_target": K_TARGET, "K_hedge": K_HEDGE, "T_total": T_TOTAL, "sigma": SIGMA, "r": R, "kappa": kappa,
        "hedge_maturity_factor": HEDGE_MATURITY_FACTOR,
    }
    save_model(model, f"{OUT_DIR}/hedgenet_nb10_n30_kappa0.01_deltaneutral.pt", fingerprint, losses)

print("done.")
