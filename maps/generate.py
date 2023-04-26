from laser_env import DynamicLaserEnv, ObservationType
import cv2


def generate():
    env = DynamicLaserEnv(
        width=11,
        height=11,
        num_agents=2,
        num_gems=5,
        num_lasers=2,
        obs_type=ObservationType.FLATTENED,
        wall_density=0.1
    )

    folder = "maps/11x11"
    for i in range(10):
        env.reset()
        image = env.render("rgb_array")
        cv2.imwrite(f"{folder}/env-{i}.png", image)
        summary = env.summary()
        with open(f"{folder}/env-{i}", "w") as f:
            f.write(summary["map_file_content"])


if __name__ == "__main__":
    generate()