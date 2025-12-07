"""
Validation and aggregation helpers for worker-submitted updates.
Integrates data integrity (block hash), norm checks, optional validation loss, and robust aggregation.
"""
import hashlib
from typing import Dict, List, Optional, Tuple

import torch


class UpdateRejected(Exception):
    """Raised when an update fails validation."""


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_block_hash(submitted_hash: str, canonical_hash: str) -> None:
    if submitted_hash != canonical_hash:
        raise UpdateRejected(f"block_hash mismatch (got {submitted_hash}, expected {canonical_hash})")


def validate_metrics(metrics: Dict, max_loss: float = 1e4) -> None:
    if not metrics:
        return
    for key in ["train_loss_mean", "train_loss_last", "grad_norm_mean"]:
        if key in metrics:
            val = metrics[key]
            if val is None or torch.isnan(torch.tensor(val)) or torch.isinf(torch.tensor(val)):
                raise UpdateRejected(f"metrics_invalid: {key} is nan/inf")
            if key.startswith("train_loss") and val > max_loss:
                raise UpdateRejected(f"metrics_invalid: {key} too large ({val})")


def delta_l2_norm(delta_state: Dict[str, torch.Tensor]) -> float:
    total = 0.0
    for t in delta_state.values():
        total += (t.float() ** 2).sum().item()
    return total**0.5


def validate_delta_norm(delta_state: Dict[str, torch.Tensor], min_norm: float, max_norm: float) -> float:
    norm = delta_l2_norm(delta_state)
    if norm < min_norm:
        raise UpdateRejected(f"delta_norm too small ({norm:.4e})")
    if norm > max_norm:
        raise UpdateRejected(f"delta_norm too large ({norm:.4e})")
    return norm


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def reference_alignment(
    delta_state: Dict[str, torch.Tensor],
    ref_state: Dict[str, torch.Tensor],
    min_cosine: float = -0.1,
) -> float:
    """
    Compare delta against a small reference gradient (same keys/shapes).
    Returns average cosine. Reject if below min_cosine.
    """
    cosines: List[float] = []
    for k, ref_tensor in ref_state.items():
        if k not in delta_state:
            continue
        cosines.append(
            cosine_similarity(delta_state[k].to(ref_tensor.device).float(), ref_tensor.float())
        )
    if not cosines:
        return 1.0
    avg_cos = sum(cosines) / len(cosines)
    if avg_cos < min_cosine:
        raise UpdateRejected(f"reference_alignment too low ({avg_cos:.3f})")
    return avg_cos


def trimmed_mean(deltas: List[Dict[str, torch.Tensor]], trim_frac: float = 0.1) -> Dict[str, torch.Tensor]:
    """
    Robust aggregation: per-parameter trimmed mean.
    """
    if not deltas:
        return {}
    keys = deltas[0].keys()
    agg: Dict[str, torch.Tensor] = {}
    lower = trim_frac
    upper = 1.0 - trim_frac
    for k in keys:
        stacked = torch.stack([d[k] for d in deltas], dim=0)
        # flatten per-tensor for sorting
        flat = stacked.view(stacked.shape[0], -1)
        # sort along batch dim
        sorted_flat, _ = torch.sort(flat, dim=0)
        low_idx = int(lower * sorted_flat.shape[0])
        high_idx = max(low_idx + 1, int(upper * sorted_flat.shape[0]))
        trimmed = sorted_flat[low_idx:high_idx].mean(dim=0)
        agg[k] = trimmed.view_as(deltas[0][k])
    return agg


def validate_update(
    delta_state: Dict[str, torch.Tensor],
    metrics: Dict,
    block_hash_submitted: Optional[str],
    block_hash_canonical: Optional[str],
    min_norm: float = 1e-6,
    max_norm: float = 1e9,
    ref_grad: Optional[Dict[str, torch.Tensor]] = None,
    min_cosine: float = -0.1,
) -> Tuple[float, Optional[float]]:
    """
    Run validation checks on a single update. Returns (norm, ref_cosine).
    Raises UpdateRejected on failure.
    """
    if block_hash_submitted and block_hash_canonical:
        validate_block_hash(block_hash_submitted, block_hash_canonical)
    validate_metrics(metrics)
    norm = validate_delta_norm(delta_state, min_norm=min_norm, max_norm=max_norm)
    ref_cos = None
    if ref_grad:
        ref_cos = reference_alignment(delta_state, ref_grad, min_cosine=min_cosine)
    return norm, ref_cos
