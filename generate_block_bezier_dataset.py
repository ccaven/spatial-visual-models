from functools import partial

from src.environment.generic import (
    SyntheticSceneConfig,
    SyntheticVideoDatasetConfig,
    generate_dataset
)

from src.environment.block_scene import (
    BlockSceneConfig,
    block_scene_factory
)

from src.environment.bezier_trajectory import (
    BezierTrajectoryConfig,
    bezier_trajectory_factory
)

if __name__ == '__main__':        
    generate_dataset(SyntheticVideoDatasetConfig(
        trajectory_factory=partial(bezier_trajectory_factory, BezierTrajectoryConfig(
            seconds_per_segment=1.0,
            num_segments=3,
            position_limit=10,
            control_point_limit=5
        )),
        scene=SyntheticSceneConfig(
            seed=587452,
            camera_fov=70,
            resolution=(128, 128),
            scene_factory=partial(block_scene_factory, config=BlockSceneConfig(
                blocks_per_face=7
            ))
        ),
        num_videos=10000,
        frames_per_video=1,
        frame_delta_time=1.0/64.0,
        save_dir="./datasets/3_25_24",
        regenerate_frequency=5
    ))