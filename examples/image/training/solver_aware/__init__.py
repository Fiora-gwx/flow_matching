from importlib import import_module


_EXPORTS = {
    "MonitorArtifacts": ("training.solver_aware.monitors", "MonitorArtifacts"),
    "SolverAwareArtifacts": ("training.solver_aware.fixed_point", "SolverAwareArtifacts"),
    "SolverAwareClockArtifacts": ("training.solver_aware.clock", "SolverAwareClockArtifacts"),
    "build_solver_aware_clock": ("training.solver_aware.clock", "build_solver_aware_clock"),
    "build_defect_clock": ("training.solver_aware.defect_clock", "build_defect_clock"),
    "build_defect_clock_profile": ("training.solver_aware.defect_clock", "build_defect_clock_profile"),
    "compute_defect_monitor": ("training.solver_aware.defect_monitor", "compute_defect_monitor"),
    "maybe_build_solver_aware_artifacts": (
        "training.solver_aware.fixed_point",
        "maybe_build_solver_aware_artifacts",
    ),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
