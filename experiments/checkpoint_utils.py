import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def find_checkpoint(
    exp_dir: Path,
    epoch: Optional[int] = None,
    warn_on_fallback: bool = True,
) -> Optional[Path]:
    if epoch is not None:
        candidates = [
            exp_dir / f"checkpoint-{epoch}.pth",
            exp_dir / f"checkpoint{epoch}.pth",
            exp_dir / f"checkpoint{epoch:04d}.pth",
        ]
        for path in candidates:
            if path.exists():
                return path
    fallback = exp_dir / "checkpoint.pth"
    if fallback.exists():
        if epoch is not None and warn_on_fallback:
            logger.warning(
                "Requested checkpoint for epoch %s was not found in %s; falling back to latest checkpoint %s",
                epoch,
                exp_dir,
                fallback,
            )
        return fallback
    return None


def resolve_checkpoint_path(
    base_dir: Path,
    spec: Dict[str, object],
    workspace_root: Optional[Path] = None,
) -> Optional[Path]:
    workspace_root = workspace_root or Path.cwd()
    explicit_path = spec.get("checkpoint_path")
    if explicit_path:
        checkpoint_path = Path(str(explicit_path))
        if not checkpoint_path.is_absolute():
            checkpoint_path = workspace_root / checkpoint_path
        return checkpoint_path if checkpoint_path.exists() else None
    exp_dir = base_dir / str(spec["dataset"]) / str(spec["name"])
    checkpoint_epoch = spec.get("checkpoint_epoch")
    return find_checkpoint(
        exp_dir=exp_dir,
        epoch=None if checkpoint_epoch is None else int(checkpoint_epoch),
    )
