import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)
CURRICULUM_SIGNATURE = "warmup0.3_linear_to1"


def _load_checkpoint_args(checkpoint_path: Path) -> Optional[Dict[str, object]]:
    args_path = checkpoint_path.parent / "args.json"
    if not args_path.exists():
        return None
    try:
        with open(args_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("Failed to read checkpoint args from %s: %s", args_path, error)
        return None
    if not isinstance(payload, dict):
        logger.warning("Checkpoint args at %s are not a JSON object.", args_path)
        return None
    return payload


def load_checkpoint_args(checkpoint_path: Path) -> Dict[str, object]:
    payload = _load_checkpoint_args(checkpoint_path)
    return {} if payload is None else payload


def checkpoint_matches_spec(
    checkpoint_path: Path,
    spec: Dict[str, object],
) -> bool:
    checkpoint_args = _load_checkpoint_args(checkpoint_path)
    return _is_semantically_compatible(spec, checkpoint_args)


def _float_matches(expected: object, observed: object, tol: float = 1e-8) -> bool:
    if expected is None:
        return True
    if observed is None:
        return False
    try:
        return abs(float(expected) - float(observed)) <= tol
    except (TypeError, ValueError):
        return False


def _normalize_strategy_semantics(payload: Dict[str, object]) -> Dict[str, object]:
    model_output_type = str(payload.get("model_output_type", "velocity"))
    default_sampling = "ds_dr_sq" if model_output_type == "base_velocity" else "uniform"
    time_sampling_strategy = str(payload.get("time_sampling_strategy", default_sampling))
    mixed_lambda = float(payload.get("mixed_lambda", 0.5))
    stratified_bins = int(payload.get("stratified_bins", 16))
    curriculum_signature = str(
        payload.get(
            "curriculum_signature",
            CURRICULUM_SIGNATURE if time_sampling_strategy == "curriculum" else "",
        )
    )
    strategy_mapping = {
        ("velocity", "uniform"): "A",
        ("base_velocity", "ds_dr_sq"): "B",
        ("velocity", "mixed_lambda"): "C",
        ("velocity", "stratified"): "D",
        ("velocity", "stratified_mixed"): "E",
        ("velocity", "curriculum"): "F",
    }
    strategy_id = strategy_mapping.get((model_output_type, time_sampling_strategy))
    if strategy_id is None:
        raise ValueError(
            "Unsupported checkpoint strategy semantics: "
            f"model_output_type={model_output_type}, "
            f"time_sampling_strategy={time_sampling_strategy}"
        )
    return {
        "model_output_type": model_output_type,
        "time_sampling_strategy": time_sampling_strategy,
        "mixed_lambda": mixed_lambda,
        "stratified_bins": stratified_bins,
        "curriculum_signature": curriculum_signature,
        "strategy_id": strategy_id,
    }


def _strategy_matches(expected_spec: Dict[str, object], checkpoint_args: Dict[str, object]) -> bool:
    expected = _normalize_strategy_semantics(expected_spec)
    observed = _normalize_strategy_semantics(checkpoint_args)
    return expected == observed


def _uses_strategy_b_semantics(
    checkpoint_args: Dict[str, object],
) -> bool:
    return (
        checkpoint_args.get("model_output_type") == "base_velocity"
        and checkpoint_args.get("time_sampling_strategy") == "ds_dr_sq"
    )


def _is_semantically_compatible(
    expected_spec: Dict[str, object],
    checkpoint_args: Optional[Dict[str, object]],
) -> bool:
    if checkpoint_args is None:
        return False

    expected_path = expected_spec.get("path_family")
    expected_clock = expected_spec.get("clock_family")
    expected_beta = expected_spec.get("clock_beta")
    observed_path = checkpoint_args.get("path_family")
    observed_clock = checkpoint_args.get("clock_family")
    observed_beta = checkpoint_args.get("clock_beta")
    observed_tag = str(checkpoint_args.get("clock_semantics_tag", ""))
    expected_tag = str(expected_spec.get("clock_semantics_tag", ""))

    if expected_path is not None and observed_path != expected_path:
        return False

    try:
        if not _strategy_matches(expected_spec, checkpoint_args):
            return False
    except ValueError:
        return False

    if expected_clock == "ft_beta":
        if observed_clock != "ft_beta" or not _float_matches(expected_beta, observed_beta):
            return False
        if expected_path == "linear":
            return (
                observed_path == "linear"
                and observed_tag == "ft_global_v2_linear_closed_form"
            )
        if expected_path == "trig_vp":
            return (
                observed_path == "trig_vp"
                and observed_tag.startswith("ft_global_v2_trig_vp_")
            )
        return observed_tag.startswith("ft_global_v2_")

    if expected_clock == "uniform" and expected_tag and observed_tag:
        if observed_tag != expected_tag:
            return False

    if expected_clock is not None and observed_clock != expected_clock:
        return False
    return _float_matches(expected_beta, observed_beta)


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

    checkpoint_path = resolve_checkpoint_path(
        base_dir=base_dir,
        spec={
            "dataset": source_dataset,
            "name": source_name,
            "checkpoint_epoch": checkpoint_epoch,
            "checkpoint_path": explicit_path,
        },
        workspace_root=workspace_root,
    )
    if checkpoint_path is None:
        return None

    checkpoint_args = _load_checkpoint_args(checkpoint_path)
    if not _is_semantically_compatible(spec, checkpoint_args):
        logger.warning(
            "Rejected reused checkpoint %s for %s because checkpoint semantics do not match the requested spec.",
            checkpoint_path,
            spec.get("name", "<unnamed>"),
        )
        return None
    return checkpoint_path
