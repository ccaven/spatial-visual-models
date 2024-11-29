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

from src.environment.fixed_step_trajectory import (
    FixedStepTrajectory,
    fixed_step_trajectory_factory
)

if __name__ == '__main__':
    num_points = 3
    delta_time = 1.0

    generate_dataset(SyntheticVideoDatasetConfig(
        trajectory_factory=partial(fixed_step_trajectory_factory, FixedStepTrajectory(
            num_points=num_points,
            seconds_per_point=delta_time,
            spread=3.0
        )),
        scene=SyntheticSceneConfig(
            seed=587452,
            camera_fov=70,
            resolution=(128, 128),
            scene_factory=partial(block_scene_factory, config=BlockSceneConfig(
                blocks_per_face=7
            ))
        ),
        num_videos=2000,
        frames_per_video=num_points,
        frame_delta_time=delta_time,
        save_dir="./datasets/3_26_24_3",
        regenerate_frequency=1
    ))