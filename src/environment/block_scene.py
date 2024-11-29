import pygfx as gfx
import numpy as np
import random
from wgpu.gui.offscreen import WgpuCanvas
from typing import NamedTuple, Callable
from functools import partial
from fastprogress import progress_bar

class BlockSceneConfig(NamedTuple):
    blocks_per_face: int

def block_scene_factory(rng: random.Random, scene: gfx.Scene, config: BlockSceneConfig):
    def rnd_col():
        return '#%02X%02X%02X' % (
            rng.randint(0, 255),
            rng.randint(0, 255),
            rng.randint(0, 255)
        )
    
    def rnd(a, b):
        return rng.random() * (b - a) + a
    
    def cube(x, y, z, w, h, d, color):
        mesh = gfx.Mesh(
            gfx.box_geometry(w, h, d),
            gfx.MeshPhongMaterial(color=color, shininess=10)
        )
        
        mesh.local.x = x
        mesh.local.y = y
        mesh.local.z = z
        
        scene.add(mesh)
    
    def point_light(x, y, z, color):
        light = gfx.PointLight(color, intensity=2)
        light.local.x = x
        light.local.y = y
        light.local.z = z
        scene.add(light)
    
    
    point_light(10, 0, -5, "#61f191")
    point_light(5, 10, 0, "#6188f1")
    point_light(0, -5, 10, "#e051bc")
    
    for ax in [0, 1, 2]:
        args = [0, 0, 0, 40, 40, 40, "#fff"]
        args[ax] = -20
        args[ax + 3] = 1
        cube(*args)
        args[ax] = 20
        cube(*args)
        
        for _ in range(config.blocks_per_face):
            args = [rnd(-20, 20), rnd(-20, 20), rnd(-20, 20), rnd(5, 10), rnd(5, 10), rnd(5, 10), rnd_col()]
            args[ax] = rnd(-20, -18)
            cube(*args)
            args[ax] = rnd(20, 18)
            cube(*args)