from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

EPS = 1e-8
METRIC_OUTPUTS = {
    "fid": ("fid",),
    "precision_recall": ("precision", "recall"),
    "inception_score": ("is_mean", "is_std"),
}


def requested_metrics(args) -> Tuple[str, ...]:
    metrics = set(args.metrics or [])
    if getattr(args, "compute_fid", False):
        metrics.add("fid")
    if not metrics:
        metrics.add("fid")
    return tuple(sorted(metrics))


def metric_output_names(metric_name: str) -> Tuple[str, ...]:
    return METRIC_OUTPUTS.get(metric_name, (metric_name,))


def prepare_inception_input(images: Tensor) -> Tensor:
    if images.dtype == torch.uint8:
        return images
    if images.is_floating_point():
        return images.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
    return images.to(torch.uint8)


def extract_inception_features(fid_metric, images: Tensor) -> Tensor:
    if not hasattr(fid_metric, "inception"):
        raise RuntimeError("FrechetInceptionDistance backend does not expose inception features.")
    features = fid_metric.inception(prepare_inception_input(images))
    if features.ndim > 2:
        features = features.reshape(features.shape[0], -1)
    return features.detach().to(torch.float32)


def _subsample(features: Tensor, max_samples: int) -> Tensor:
    if features.shape[0] <= max_samples:
        return features
    generator = torch.Generator(device=features.device)
    generator.manual_seed(0)
    indices = torch.randperm(features.shape[0], generator=generator, device=features.device)[:max_samples]
    return features.index_select(0, indices)


def _kth_neighbor_radius(features: Tensor, neighbors: int, chunk_size: int = 512) -> Tensor:
    radii = []
    total = features.shape[0]
    neighbors = min(neighbors + 1, total)
    for start in range(0, total, chunk_size):
        chunk = features[start : start + chunk_size]
        distances = torch.cdist(chunk, features)
        local_indices = torch.arange(chunk.shape[0], device=features.device) + start
        distances[torch.arange(chunk.shape[0], device=features.device), local_indices] = torch.inf
        radii.append(torch.kthvalue(distances, neighbors, dim=1).values)
    return torch.cat(radii, dim=0)


def _membership_fraction(
    query_features: Tensor,
    reference_features: Tensor,
    reference_radii: Tensor,
    chunk_size: int = 512,
) -> Tensor:
    hits = []
    for start in range(0, query_features.shape[0], chunk_size):
        query_chunk = query_features[start : start + chunk_size]
        membership = torch.zeros(query_chunk.shape[0], dtype=torch.bool, device=query_features.device)
        for ref_start in range(0, reference_features.shape[0], chunk_size):
            reference_chunk = reference_features[ref_start : ref_start + chunk_size]
            radii_chunk = reference_radii[ref_start : ref_start + chunk_size]
            distances = torch.cdist(query_chunk, reference_chunk)
            membership |= (distances <= radii_chunk.unsqueeze(0)).any(dim=1)
        hits.append(membership)
    return torch.cat(hits, dim=0).to(torch.float32).mean()


def compute_precision_recall(
    real_features: Tensor,
    fake_features: Tensor,
    neighbors: int,
    max_samples: int,
) -> Dict[str, float]:
    if real_features.numel() == 0 or fake_features.numel() == 0:
        return {"precision": float("nan"), "recall": float("nan")}

    device = real_features.device
    real = _subsample(real_features.to(device=device, dtype=torch.float32), max_samples=max_samples)
    fake = _subsample(fake_features.to(device=device, dtype=torch.float32), max_samples=max_samples)

    real_radii = _kth_neighbor_radius(real, neighbors=neighbors)
    fake_radii = _kth_neighbor_radius(fake, neighbors=neighbors)
    precision = _membership_fraction(fake, real, real_radii)
    recall = _membership_fraction(real, fake, fake_radii)
    return {"precision": float(precision.cpu()), "recall": float(recall.cpu())}
