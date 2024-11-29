import numpy as np
import pylinalg as la
from typing import NamedTuple
from functools import partial

class RandomPlaneTrajectory(NamedTuple):
    seconds_per_point: float
    num_points: int = 3
    spread: float = 5.0

def sample_random_plane_trajectory_at_time(t: float, positions: np.ndarray, rotations: np.ndarray, config: RandomPlaneTrajectory):
    k = round(t / config.seconds_per_point)
    mat = np.array(rotations[k])
    mat[:-1, -1] = positions[k]
    return mat

def random_plane_trajectory_factory(config: RandomPlaneTrajectory):
    target = np.array([0, 0, 35])
    up = np.array([0, 1, 0])
    positions = np.zeros((config.num_points, 3))
    
    # Randomly distribute along XY plane
    positions[:, 0:2] = np.random.rand(config.num_points, 2) * config.spread * 2 - config.spread
    
    # Fixed z position
    positions[:, 2] = 5.0

    # Focus on center point
    rotations = np.zeros((config.num_points, 4, 4))
    for i in range(config.num_points):
        rotations[i] = la.mat_look_at(positions[i], target, up)
    
    return partial(
        sample_random_plane_trajectory_at_time,
        positions=positions,
        rotations=rotations,
        config=config
    )