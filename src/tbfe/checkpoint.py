"""Safe, explicit model-weight loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch


def _unwrap_state_dict(payload: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint must contain a mapping of parameter names to tensors")
    for key in ("state_dict", "model_state_dict", "model"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            payload = candidate
            break
    if not payload or not all(isinstance(key, str) for key in payload):
        raise ValueError("could not find a valid state_dict in checkpoint")
    return payload  # type: ignore[return-value]


def _remove_common_prefix(state: Mapping[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if state and all(key.startswith(prefix) for key in state):
        return {key[len(prefix) :]: value for key, value in state.items()}
    return dict(state)


def read_state_dict(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    """Read safetensors, a plain state dict, or a Lightning checkpoint.

    PyTorch checkpoints are pickle-based. Only load files obtained from a
    trusted source; ``weights_only=True`` is requested whenever supported.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install tbfe[safetensors] to read this checkpoint") from exc
        state = load_file(str(path), device=str(map_location))
    else:
        try:
            payload = torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:  # PyTorch before weights_only support
            payload = torch.load(path, map_location=map_location)
        state = dict(_unwrap_state_dict(payload))
    for prefix in ("module.", "model.", "_orig_mod."):
        state = _remove_common_prefix(state, prefix)
    return state


def load_model_weights(
    model: torch.nn.Module,
    path: str | Path,
    *,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> tuple[list[str], list[str]]:
    state = read_state_dict(path, map_location=map_location)
    target_keys = set(model.state_dict())
    state_keys = set(state)
    if state_keys and not (state_keys & target_keys):
        with_network_prefix = {f"network.{key}": value for key, value in state.items()}
        without_network_prefix = {
            key[len("network.") :]: value
            for key, value in state.items()
            if key.startswith("network.")
        }
        if set(with_network_prefix) & target_keys:
            state = with_network_prefix
        elif set(without_network_prefix) & target_keys:
            state = without_network_prefix
    incompatible = model.load_state_dict(state, strict=strict)
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)
