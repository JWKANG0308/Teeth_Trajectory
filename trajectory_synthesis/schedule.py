# src/schedule.py
import numpy as np
from dataclasses import dataclass
from .utils import _safe_int_key


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


@dataclass
class ScheduleConfig:
    start_max: float = 0.35
    gamma_min: float = 0.8
    gamma_max: float = 1.4
    group_mode: str = "posterior_to_anterior"  # "none" or "posterior_to_anterior"
    group_bonus: float = 0.08
    seed: int = 42


def tooth_group_posterior_to_anterior(tooth_id: str):
    # FDI ones digit: 1 incisor ... 7/8 molar
    try:
        tid = int(tooth_id)
        ones = tid % 10
        max_ones = 8
        g = max_ones - ones
        return int(np.clip(g, 0, 7))
    except Exception:
        return 0


def sample_schedule_params(tooth_ids, cfg: ScheduleConfig, rng: np.random.Generator):
    start = {}
    gamma = {}
    for tid in tooth_ids:
        start[tid] = float(rng.uniform(0.0, cfg.start_max))
        gamma[tid] = float(rng.uniform(cfg.gamma_min, cfg.gamma_max))

    if cfg.group_mode == "posterior_to_anterior":
        for tid in tooth_ids:
            g = tooth_group_posterior_to_anterior(tid)
            start[tid] = float(np.clip(start[tid] + cfg.group_bonus * (g / 7.0), 0.0, 0.95))
    return start, gamma


def progress_with_delay_and_warp(s_global: float, start_delay: float, gamma: float):
    p = (s_global - start_delay) / max(1e-6, (1.0 - start_delay))
    p = float(np.clip(p, 0.0, 1.0))
    p = float(smoothstep(p) ** gamma)
    return p
