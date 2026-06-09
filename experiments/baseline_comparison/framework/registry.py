"""
registry.py — name -> Detector factory.

Detectors register a factory `cfg -> Detector`. The runner asks the registry to
build only the detectors named in the config, passing each its sub-config.
"""

from __future__ import annotations

from typing import Callable, Dict, List
from .detector_api import Detector

_REGISTRY: Dict[str, Callable[[dict], Detector]] = {}


def register(name: str):
    """Decorator: register a Detector subclass under `name`."""
    def _wrap(cls):
        if name in _REGISTRY:
            raise ValueError(f"Detector '{name}' already registered")
        _REGISTRY[name] = lambda cfg=None, _cls=cls: _cls(cfg)
        cls.name = name
        return cls
    return _wrap


def build(name: str, cfg: dict | None = None) -> Detector:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown detector '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](cfg)


def available() -> List[str]:
    return sorted(_REGISTRY)


def ensure_loaded() -> None:
    """Import the detectors package so all @register side-effects run."""
    import experiments.baseline_comparison.detectors  # noqa: F401
