from typing import Iterable, List, Sequence


LOW_NFE_FOCUS_MAX = 20.0
LOW_NFE_TAIL_COMPRESSION = 5.0


def transform_focus_axis_value(
    value: float,
    focus_max: float = LOW_NFE_FOCUS_MAX,
    tail_compression: float = LOW_NFE_TAIL_COMPRESSION,
) -> float:
    value = float(value)
    if value <= focus_max:
        return value
    return focus_max + (value - focus_max) / tail_compression


def transform_focus_axis_values(
    values: Iterable[float],
    focus_max: float = LOW_NFE_FOCUS_MAX,
    tail_compression: float = LOW_NFE_TAIL_COMPRESSION,
) -> List[float]:
    return [
        transform_focus_axis_value(
            value,
            focus_max=focus_max,
            tail_compression=tail_compression,
        )
        for value in values
    ]


def selected_nfe_ticks(
    values: Sequence[int],
    focus_max: float = LOW_NFE_FOCUS_MAX,
) -> List[int]:
    unique_values = sorted({int(value) for value in values})
    focus_ticks = [value for value in unique_values if value <= focus_max]
    tail_ticks = [value for value in unique_values if value > focus_max]
    if len(tail_ticks) > 4:
        tail_ticks = [tail_ticks[0], tail_ticks[1], tail_ticks[-2], tail_ticks[-1]]
    return focus_ticks + tail_ticks
