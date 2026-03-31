from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor


STORK_DEFAULT_S = 9


@dataclass
class STORKState:
    step_index: int = 0
    last_velocity: Optional[Tensor] = None
    last_dt: Optional[float] = None
    virtual_stage_count: int = 0
    s: int = STORK_DEFAULT_S


@lru_cache(maxsize=1)
def load_stork_coefficients() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coefficients_path = Path(__file__).with_name("stork_constants.npz")
    if not coefficients_path.exists():
        raise FileNotFoundError(
            f"Missing STORK coefficient table at {coefficients_path}. "
            "Regenerate examples/image/training/stork_constants.npz before using stork4."
        )
    with np.load(coefficients_path, allow_pickle=False) as data:
        ms = data["ms"].astype(np.int64, copy=False)
        fpa = data["fpa"].astype(np.float32, copy=False)
        fpb = data["fpb"].astype(np.float32, copy=False)
        recf = data["recf"].astype(np.float32, copy=False)
    return ms, fpa, fpb, recf


def mdegr(requested_degree: int, available_degrees: np.ndarray) -> Tuple[int, int, int]:
    offsets = []
    offset = 0
    for degree in available_degrees:
        offsets.append(offset)
        offset += int(degree) * 2 - 1

    for index, degree in enumerate(available_degrees):
        degree_int = int(degree)
        if degree_int >= requested_degree:
            return degree_int, index, offsets[index]

    last_index = len(available_degrees) - 1
    return int(available_degrees[last_index]), last_index, offsets[last_index]


def taylor_approximation(
    order: int,
    diff: float,
    model_output: Tensor,
    derivative: Tensor,
    second_derivative: Optional[Tensor] = None,
    third_derivative: Optional[Tensor] = None,
) -> Tensor:
    diff_tensor = torch.as_tensor(
        diff,
        device=model_output.device,
        dtype=model_output.dtype,
    )
    if order == 1:
        return model_output + diff_tensor * derivative
    if order == 2:
        if second_derivative is None:
            raise ValueError("second_derivative is required for second-order Taylor approximation.")
        return model_output + diff_tensor * derivative + 0.5 * diff_tensor.square() * second_derivative
    if order == 3:
        if second_derivative is None or third_derivative is None:
            raise ValueError("Second and third derivatives are required for third-order Taylor approximation.")
        return (
            model_output
            + diff_tensor * derivative
            + 0.5 * diff_tensor.square() * second_derivative
            + diff_tensor.pow(3) * third_derivative / 6.0
        )
    raise ValueError(f"Unsupported Taylor approximation order={order}.")


def stork4_step(
    model_output: Tensor,
    sample: Tensor,
    t_start: float,
    t_end: float,
    state: STORKState,
) -> Tensor:
    """Advance one STORK4 super-step on an arbitrary local interval [t_start, t_end].

    Phase-1 only claims non-uniform node support plus a first-order Taylor
    virtual-stage approximation. The actual network evaluation is the cached
    model_output at t_start; all internal stages reuse the current local step
    size h_n = t_end - t_start.
    """
    step_dt = float(t_end - t_start)
    if step_dt <= 0.0:
        raise ValueError(f"stork4 requires a positive step size. Got t_start={t_start}, t_end={t_end}.")

    if state.step_index == 0:
        next_sample = sample + step_dt * model_output
        state.last_velocity = model_output.detach().clone()
        state.last_dt = step_dt
        state.step_index += 1
        return next_sample

    if state.last_velocity is None or state.last_dt is None:
        raise RuntimeError("stork4 state is missing startup history.")

    velocity_derivative = (model_output - state.last_velocity) / float(state.last_dt)
    state.last_velocity = model_output.detach().clone()
    state.last_dt = step_dt
    state.step_index += 1

    ms, fpa, fpb, recf = load_stork_coefficients()
    mdeg, degree_index, recf_offset = mdegr(state.s, ms)

    y_prev_prev = sample
    y_prev = sample
    y_current = sample
    c_prev_prev = float(t_start)
    c_prev = float(t_start)
    c_current = float(t_start)

    for j in range(1, mdeg + 1):
        if j == 1:
            advance = step_dt * float(recf[recf_offset])
            c_current = t_start + advance
            c_prev = c_current
            y_current = sample + advance * model_output
            y_prev = y_current
            continue

        diff = c_current - t_start
        velocity = taylor_approximation(
            order=1,
            diff=diff,
            model_output=model_output,
            derivative=velocity_derivative,
        )
        advance = step_dt * float(recf[recf_offset + 2 * (j - 2) + 1])
        mix_prev_prev = -float(recf[recf_offset + 2 * (j - 2) + 2])
        mix_prev = 1.0 - mix_prev_prev

        next_time = advance + mix_prev * c_prev + mix_prev_prev * c_prev_prev
        next_sample = advance * velocity + mix_prev * y_prev + mix_prev_prev * y_prev_prev

        y_prev_prev = y_prev
        y_prev = next_sample
        y_current = next_sample
        c_prev_prev = c_prev
        c_prev = next_time
        c_current = next_time

    finish_1 = step_dt * float(fpa[degree_index, 0])
    velocity_1 = taylor_approximation(
        order=1,
        diff=c_current - t_start,
        model_output=model_output,
        derivative=velocity_derivative,
    )

    c_finish = c_current + finish_1
    finish_2a = step_dt * float(fpa[degree_index, 1])
    finish_2b = step_dt * float(fpa[degree_index, 2])
    velocity_2 = taylor_approximation(
        order=1,
        diff=c_finish - t_start,
        model_output=model_output,
        derivative=velocity_derivative,
    )

    c_finish = c_current + finish_2a + finish_2b
    finish_3a = step_dt * float(fpa[degree_index, 3])
    finish_3b = step_dt * float(fpa[degree_index, 4])
    finish_3c = step_dt * float(fpa[degree_index, 5])
    velocity_3 = taylor_approximation(
        order=1,
        diff=c_finish - t_start,
        model_output=model_output,
        derivative=velocity_derivative,
    )

    c_finish = c_current + finish_3a + finish_3b + finish_3c
    finish_4a = step_dt * float(fpb[degree_index, 0])
    finish_4b = step_dt * float(fpb[degree_index, 1])
    finish_4c = step_dt * float(fpb[degree_index, 2])
    finish_4d = step_dt * float(fpb[degree_index, 3])
    velocity_4 = taylor_approximation(
        order=1,
        diff=c_finish - t_start,
        model_output=model_output,
        derivative=velocity_derivative,
    )

    state.virtual_stage_count += max(0, mdeg - 1) + 4
    return (
        y_current
        + finish_4a * velocity_1
        + finish_4b * velocity_2
        + finish_4c * velocity_3
        + finish_4d * velocity_4
    )
