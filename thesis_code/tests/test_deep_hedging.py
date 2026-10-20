import runpy

import pytest
import torch

from src import config
from src.deep_hedging import HedgeNet, RecurrentHedgeNet, RecurrentTwoAssetHedgeNet, TwoAssetHedgeNet, load_model

SHIPPED = {  # checkpoint file in models/ -> network class
    "hedgenet_nb01_entropic_l1.pt": HedgeNet,
    "hedgenet_nb07_mse_3feat.pt": TwoAssetHedgeNet,
    "hedgenet_nb05_entropic_kappa1e-02.pt": RecurrentHedgeNet,
    "hedgenet_kappa_sweep_1e-02.pt": RecurrentTwoAssetHedgeNet,
    "hedgenet_kappa_sweep_3e-03.pt": RecurrentTwoAssetHedgeNet,
    "hedgenet_kappa_sweep_1e-03.pt": RecurrentTwoAssetHedgeNet,
    "hedgenet_kappa_sweep_3e-04.pt": RecurrentTwoAssetHedgeNet,
    "hedgenet_kappa_sweep_1e-04.pt": RecurrentTwoAssetHedgeNet,
    "hedgenet_nb09_n120_kappa0.01.pt": RecurrentTwoAssetHedgeNet,
    "hedgenet_nb10_n30_kappa0.01_deltaneutral.pt": RecurrentTwoAssetHedgeNet,
}


def run_train_script(tmp_path, old_line, new_line):
    # a copy of scripts/train.py with one top line replaced, run in tmp_path
    source = (config.ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    assert source.count(old_line) == 1
    (tmp_path / "train.py").write_text(source.replace(old_line, new_line), encoding="utf-8")
    runpy.run_path(str(tmp_path / "train.py"))


def test_shipped_networks_load():
    assert sorted(p.name for p in config.MODELS_DIR.glob("*.pt")) == sorted(SHIPPED)
    for name, network_class in SHIPPED.items():
        model = load_model(config.MODELS_DIR / name)
        assert type(model) is network_class
        assert not model.training


def test_train_script_refuses_models_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        run_train_script(tmp_path, 'OUT_DIR = "models_retrained"', 'OUT_DIR = "models"')
    assert not (tmp_path / "models").exists()


def test_train_script_two_epochs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_train_script(tmp_path, "N_EPOCHS_OVERRIDE = None", "N_EPOCHS_OVERRIDE = 2")
    out = tmp_path / "models_retrained"
    assert sorted(p.name for p in out.glob("*.pt")) == sorted(SHIPPED)
    for name, network_class in SHIPPED.items():
        shipped = torch.load(config.MODELS_DIR / name, weights_only=True)
        retrained = torch.load(out / name, weights_only=True)
        # train.py stores the settings of the shipped network; only the number of epochs differs
        assert retrained["fingerprint"] == dict(shipped["fingerprint"], n_epochs=2)
        assert len(retrained["loss_history"]) == 2
        assert type(load_model(out / name)) is network_class

    # the shipped delta-neutral checkpoint kept its training losses: the retrain starts the same way
    name = "hedgenet_nb10_n30_kappa0.01_deltaneutral.pt"
    shipped_losses = torch.load(config.MODELS_DIR / name, weights_only=True)["loss_history"]
    retrained_losses = torch.load(out / name, weights_only=True)["loss_history"]
    assert retrained_losses == pytest.approx(shipped_losses[:2], rel=1e-6)
