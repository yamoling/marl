from laser_env import DynamicLaserEnv, ObservationType
import json
import cv2


env = DynamicLaserEnv(
    width=5,
    height=5,
    num_lasers=1,
    num_agents=2,
    num_gems=5,
    obs_type=ObservationType.FLATTENED,
    wall_density=0.15
)
import os
os.makedirs("envs-test", exist_ok=True)
for i in range(10):
    env.reset()
    with open(f"envs-test/env-{i}.json", "w") as f:
        summary = env.summary()
        json.dump(summary, f)

    # Save one image
    img = env.render("rgb_array")
    cv2.imwrite(f"envs-test/env-{i}.png", img)