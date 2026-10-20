"""Market parameters, seeds and path counts shared by the three scripts in scripts/ and by src/."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

# Market
S0 = 100
MU = 0.0
SIGMA = 0.2
R = 0.0
T_TOTAL = 30 / 365
N_STEPS = 30
N_STEPS_FINE = 120
K_SINGLE = 100
K_TARGET = 100
K_HEDGE = 105
HEDGE_MATURITY_FACTOR = 1.25

# Paths
N_TRAIN = 100_000
N_TEST = 25_000
SEED_TRAIN_PATHS = 111
SEED_TEST_PATHS = 999

# Risk aversion lambda: rho in the evaluation and the loss of the two-asset networks with costs
RISK_AVERSION = 2.0

# Gamma-neutral benchmark and the cross-check of the Rothert positions against the network P&L chain
GN_CLIP_LAMBDA = 20.0
GN_EPS = 1e-8
CROSS_CHECK_RTOL = 1e-5
CROSS_CHECK_ATOL = 1e-5
