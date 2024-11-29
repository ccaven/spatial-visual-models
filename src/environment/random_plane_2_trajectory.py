import numpy as np
import pylinalg as la
from typing import NamedTuple
from functools import partial

class RandomPlane2Trajectory(NamedTuple):
    seconds_per_point: float
    num_points: int = 3
    spread: float = 5.0

def sample_random_plane_2_trajectory_at_time(t: float, positions: np.ndarray, rotations: np.ndarray, config: RandomPlane2Trajectory):
    k = round(t / config.seconds_per_point)
    mat = np.array(rotations[k])
    mat[:-1, -1] = positions[k]
    return mat

def random_plane_2_trajectory_factory(config: RandomPlane2Trajectory):
    up = np.array([0, 1, 0])

    # Distribute positions along XY plane
    positions = np.zeros((config.num_points, 3))
    positions[:, 0:2] = np.random.rand(config.num_points, 2) * config.spread * 2 - config.spread
    positions[:, 2] = 5.0

    targets = np.zeros((config.num_points, 3))
    
    # Distribute targets along opposite XY plane
    targets[:, 0:2] = np.random.rand(config.num_points, 2) * 30 - 15
    targets[:, 2] = 20.0
    rotations = np.zeros((config.num_points, 4, 4))
    for i in range(config.num_points):
        rotations[i] = la.mat_look_at(positions[i], targets[i], up)
    
    return partial(
        sample_random_plane_2_trajectory_at_time,
        positions=positions,
        rotations=rotations,
        config=config
    )