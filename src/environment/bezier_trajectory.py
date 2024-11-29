import pygfx as gfx
import numpy as np
import random
from wgpu.gui.offscreen import WgpuCanvas
from typing import NamedTuple, Callable
from functools import partial
from fastprogress import progress_bar

import pylinalg as la
import quaternion as np_quat

def matrix_from_position_and_quaternion(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    output_matrix = la.mat_from_quat(q)
    output_matrix[:-1, -1] = p
    return output_matrix

def smoothstep(t: float) -> float:
    return t * t * (3 - 2 * t)

class BezierTrajectoryConfig(NamedTuple):
    seconds_per_segment: float
    num_segments: int
    position_limit: float
    control_point_limit: float

def bezier_trajectory_sample_at_time(t: float, positions: np.ndarray, control: np.ndarray, quaternions: np.ndarray, config: BezierTrajectoryConfig):
    
    k = t / config.seconds_per_segment
    
    i = int(k)
    
    if i == k:
        return matrix_from_position_and_quaternion(
            positions[i],
            quaternions[i]
        )
    
    j = i + 1
    k = k - i
    
    a = positions[i]
    b = positions[j]
    c1 = control[i]
    c2 = b * 2 - control[j]
    
    # Cubic bezier curve
    coef1 = (1 - k) ** 3
    coef2 = (1 - k) ** 2 * k * 3
    coef3 = (1 - k) ** 1 * k ** 2 * 3
    coef4 = k ** 3
    
    position = coef1 * a + coef2 * c1 + coef3 * c2 + coef4 * b
    
    lerped_rotation = np_quat.as_float_array(
        np_quat.slerp(
            np_quat.as_quat_array(quaternions[i]),
            np_quat.as_quat_array(quaternions[j]),
            0,
            1,
            smoothstep(k)
        )
    )

    return matrix_from_position_and_quaternion(
        position,
        lerped_rotation
    )

def bezier_trajectory_factory(config: BezierTrajectoryConfig):
    positions = np.random.rand(config.num_segments, 3) * config.position_limit * 2 - config.position_limit
    control = np.random.rand(config.num_segments, 3) * config.control_point_limit * 2 - config.control_point_limit
    quaternions = np.random.randn(2, 4)
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    return partial(bezier_trajectory_sample_at_time, positions=positions, control=control,quaternions=quaternions, config=config)
