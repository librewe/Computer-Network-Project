"""Runtime configuration helpers backed by config.yaml."""

from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_runtime_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_runtime_model_path():
    config = load_runtime_config()
    return str((CONFIG_PATH.parent / config["model"]["path"]).resolve())
