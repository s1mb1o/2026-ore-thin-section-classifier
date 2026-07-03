from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .component_analysis import ComponentRuleConfig

RULE_CONFIG_FIELDS = (
    "fine_dark_inside_ratio",
    "fine_solidity_max",
    "fine_compactness_max",
    "talc_fraction_threshold",
)


def default_rule_config() -> dict[str, float]:
    defaults = asdict(ComponentRuleConfig())
    return {key: float(defaults[key]) for key in RULE_CONFIG_FIELDS}


def add_rule_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--rule-config-json",
        type=Path,
        default=None,
        help="Optional calibration JSON. Accepts either a direct config object or scripts/calibrate_ore_rules.py output.",
    )
    parser.add_argument("--fine-dark-inside-ratio", type=float, default=None)
    parser.add_argument("--fine-solidity-max", type=float, default=None)
    parser.add_argument("--fine-compactness-max", type=float, default=None)
    parser.add_argument("--talc-fraction-threshold", type=float, default=None)


def resolve_rule_config_from_args(args: argparse.Namespace) -> dict[str, float]:
    config = default_rule_config()
    if getattr(args, "rule_config_json", None) is not None:
        config.update(load_rule_config(Path(args.rule_config_json)))
    for key in RULE_CONFIG_FIELDS:
        cli_value = getattr(args, key, None)
        if cli_value is not None:
            config[key] = float(cli_value)
    return validate_rule_config(config)


def load_rule_config(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_config: dict[str, Any]
    if isinstance(payload, dict) and isinstance(payload.get("best_config"), dict):
        raw_config = payload["best_config"]
    elif isinstance(payload, dict) and isinstance(payload.get("config"), dict):
        raw_config = payload["config"]
    elif isinstance(payload, dict):
        raw_config = payload
    else:
        raise ValueError(f"Rule config JSON must contain an object: {path}")

    config = {key: float(raw_config[key]) for key in RULE_CONFIG_FIELDS if key in raw_config}
    if not config:
        expected = ", ".join(RULE_CONFIG_FIELDS)
        raise ValueError(f"Rule config JSON has no known fields ({expected}): {path}")
    return config


def rule_config_cli_args(config: dict[str, float]) -> list[str]:
    args: list[str] = []
    for key in RULE_CONFIG_FIELDS:
        args.extend([f"--{key.replace('_', '-')}", str(float(config[key]))])
    return args


def validate_rule_config(config: dict[str, float]) -> dict[str, float]:
    missing = [key for key in RULE_CONFIG_FIELDS if key not in config]
    if missing:
        raise ValueError(f"Rule config missing required fields: {', '.join(missing)}")
    for key in RULE_CONFIG_FIELDS:
        value = float(config[key])
        if value < 0:
            raise ValueError(f"{key} must be non-negative")
        config[key] = value
    return config
