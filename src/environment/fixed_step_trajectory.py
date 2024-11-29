import numpy as np
import pylinalg as la
from typing import NamedTuple
from functools import partial


class FixedStepTrajectory(NamedTuple):
    seconds_per_point: float
    num_points: int = 3
    spread: float = 5.0

def sample_fixed_step_trajectory_at_time(t: float, positions: np.ndarray, rotations: np.ndarray, config: FixedStepTrajectory):
    k = round(t / config.seconds_per_point)
    mat = np.array(rotations[k])
    mat[:-1, -1] = positions[k]
    return mat

def fixed_step_trajectory_factory(config: FixedStepTrajectory):
    target = np.array([0, 0, 35])
    up = np.array([0, 1, 0])
    positions = np.zeros((config.num_points, 3))
    positions[:, 0] = np.linspace(-1.0, 1.0, config.num_points) * config.spread
    positions[:, 2] = 5.0

    rotations = np.zeros((config.num_points, 4, 4))
    for i in range(config.num_points):
        rotations[i] = la.mat_look_at(positions[i], target, up)
    
    return partial(
        sample_fixed_step_trajectory_at_time,
        positions=positions,
        rotations=rotations,
        config=config
    )