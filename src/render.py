import laser_env
import cv2

for level in ["lvl1", "lvl2", "lvl3", "lvl4", "lvl5"]:
    env = laser_env.StaticLaserEnv(f"maps/normal/{level}")
    env.reset()
    img = env.render("rgb_array")

    cv2.imwrite(f"{level}.png", img)
