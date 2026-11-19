"""Конфиг опыта через CURV_CONFIG: свой файл колец, основной не трогается."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROBE = "from src.data import RINGS_FILE; from src.config import CONFIG_PATH; print(RINGS_FILE, CONFIG_PATH.name)"


def run(env_extra):
    env = {k: v for k, v in os.environ.items() if k != "CURV_CONFIG"} | env_extra
    out = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    return out.stdout.split()


def test_default_config_keeps_rings_file():
    assert run({}) == ["rings.npz", "default.yaml"]


def test_experiment_config_uses_own_rings_file(tmp_path):
    exp = tmp_path / "exp.yaml"
    shutil.copy(ROOT / "configs" / "default.yaml", exp)
    assert run({"CURV_CONFIG": str(exp)}) == ["rings.exp.npz", "exp.yaml"]
