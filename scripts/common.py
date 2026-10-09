"""Shared helpers for the METABRIC survival-aware multi-omic pipeline.

Every stage script imports this module for config loading, logging, and the
small set of I/O / survival-label parsing routines that are identical across
stages. Nothing here is stage-specific.
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_git_commit(cwd: str | Path = ".") -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "no-git-commit"


def get_logger(name: str, log_dir: str | Path) -> logging.Logger:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    fh = logging.FileHandler(log_dir / f"{name}.log", mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


def parse_os_status(series: pd.Series) -> pd.Series:
    """'0:LIVING' / '1:DECEASED' -> integer event indicator (1 = death)."""
    return series.astype(str).str.extract(r"^(\d)")[0].astype("Int64")


def parse_rfs_status(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"^(\d)")[0].astype("Int64")


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve(path_str: str) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(path_str)
    return p if p.is_absolute() else project_root() / p
