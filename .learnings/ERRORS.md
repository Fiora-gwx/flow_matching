## [ERR-20260323-001] unittest_shared_fake_matplotlib

**Logged**: 2026-03-23T00:00:00+08:00
**Priority**: medium
**Status**: pending
**Area**: tests

### Summary
Running multiple visualization-related unittest modules in one Python process can fail because fake `matplotlib` modules installed by one test leave `__spec__ = None` for later imports.

### Error
```
ValueError: matplotlib.__spec__ is None
```

### Context
- Command attempted: `python3 -m unittest tests.test_continuous_runtime tests.test_result_utils tests.test_run_experiments tests.test_visualize_results tests.test_visualize_solver_sensitivity tests.test_particle_trajectory_plot tests.test_sampling_progression_plot tests.test_analyze_mechanisms`
- Failure surfaced while importing `tests/test_particle_trajectory_plot.py`
- Root cause is cross-test process contamination, not necessarily a runtime regression

### Suggested Fix
Run visualization tests in isolated Python processes or give the fake matplotlib module a valid `__spec__`.

### Metadata
- Reproducible: yes
- Related Files: tests/test_visualize_results.py, tests/test_visualize_solver_sensitivity.py, tests/test_particle_trajectory_plot.py, tests/test_sampling_progression_plot.py

---
