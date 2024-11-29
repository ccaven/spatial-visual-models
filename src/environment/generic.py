import pygfx as gfx
import numpy as np
import random
from wgpu.gui.offscreen import WgpuCanvas
from typing import NamedTuple, Callable
from fastprogress import progress_bar
import pickle

""" Generic implementation of scene, dataset """

class SyntheticSceneConfig(NamedTuple):
    scene_factory: Callable[[random.Random, gfx.Scene], None]
    seed: int
    camera_fov: int
    resolution: tuple[int, int]

class SyntheticScene:
    def __init__(self, config: SyntheticSceneConfig):
        self.config = config
        self.camera = gfx.PerspectiveCamera(
            fov=config.camera_fov, 
            aspect=config.resolution[0]/config.resolution[1],
            depth_range=(0.2, 100)
        )
        self.scene = gfx.Scene()
        self.canvas = WgpuCanvas(
            size=config.resolution, 
            pixel_ratio=1
        )
        self.renderer = gfx.renderers.WgpuRenderer(self.canvas)
        self.rng = random.Random(config.seed)
        
        self.config.scene_factory(self.rng, self.scene)

    def reconstruct(self):
        self.scene.clear()
        self.config.scene_factory(self.rng, self.scene)
    
    def render(self) -> np.ndarray:
        self.renderer.render(self.scene, self.camera)
        render = np.asarray(self.canvas.draw())
        render = np.delete(render, 3, 2) # Remove alpha channel
        return render

    def render_at(self, matrix: np.ndarray) -> np.ndarray:
        self.camera.local.matrix = matrix
        return self.render()

class SyntheticVideoDatasetConfig(NamedTuple):
    trajectory_factory: Callable[[], Callable[[float], np.ndarray]]
    scene: SyntheticSceneConfig
    num_videos: int
    frames_per_video: int
    frame_delta_time: int
    save_dir: str
    regenerate_frequency: int | None

def generate_dataset(config: SyntheticVideoDatasetConfig):
    scene = SyntheticScene(config.scene)
    
    all_frames = []
    all_matrices = []

    for i in progress_bar(range(config.num_videos)):
        if config.regenerate_frequency is not None and i > 0 and i % config.regenerate_frequency == 0:
            scene.reconstruct()
        
        trajectory = config.trajectory_factory()
        
        matrices = []
        frames = []
        for j in range(config.frames_per_video):
            matrix = trajectory(j * config.frame_delta_time)
            frame = scene.render_at(matrix)
            matrices.append(matrix)
            frames.append(frame)
        
        matrices = np.stack(matrices, axis=0)
        frames = np.stack(frames, axis=0)
    
        all_matrices.append(matrices)
        all_frames.append(frames)
    
    all_matrices = np.stack(all_matrices, axis=0)
    all_frames = np.stack(all_frames, axis=0)
    
    np.save(config.save_dir + "/matrices.npy", all_matrices)
    np.save(config.save_dir + "/frames.npy", all_frames)

    with open(config.save_dir + "/config.pkl", "wb") as handle:
        pickle.dump(config, handle)