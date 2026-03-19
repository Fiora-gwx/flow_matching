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


def _format_template_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return format(value, "g").replace(".", "_")
    return str(value).replace(".", "_")


def _render_reference_template(template: str, spec: Dict[str, object]) -> str:
    context = {key: value for key, value in spec.items()}
    context["clock_beta_tag"] = _format_template_value(spec.get("clock_beta"))
    rendered = template
    for key, value in context.items():
        rendered = rendered.replace("{" + key + "}", str(value))
        rendered = rendered.replace("{" + key + "_tag}", _format_template_value(value))
    return rendered


def resolve_reused_checkpoint(
    reference: Dict[str, object],
    spec: Dict[str, object],
    workspace_root: Optional[Path] = None,
) -> Optional[Path]:
    workspace_root = workspace_root or Path.cwd()
    artifact_group = reference.get("artifact_group")
    if artifact_group:
        base_dir = workspace_root / "experiments" / "results" / str(artifact_group)
    else:
        base_dir = workspace_root / "experiments" / "results"

    source_name = str(
        reference.get("source_exp_name")
        or reference.get("source_name_template")
        or spec["name"]
    )
    source_name = _render_reference_template(source_name, spec)
    source_dataset = str(reference.get("dataset", spec["dataset"]))
    checkpoint_epoch = reference.get("checkpoint_epoch")
    explicit_path = reference.get("checkpoint_path")

    return resolve_checkpoint_path(
        base_dir=base_dir,
        spec={
            "dataset": source_dataset,
            "name": source_name,
            "checkpoint_epoch": checkpoint_epoch,
            "checkpoint_path": explicit_path,
        },
        workspace_root=workspace_root,
    )
