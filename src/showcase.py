import cv2
from laser_env import Action
import marl

experiment = marl.Experiment.load("logs/test")
experiment.algo.load("logs/test/run_1684317730.170918/test/160000")
experiment.algo.before_tests(0)
env = experiment.env
obs = env.reset()
done = False
i = 0
while not done:
    img = env.render("rgb_array")
    cv2.imwrite(f"step_{i}.png", img)
    action = experiment.algo.choose_action(obs)
    obs, r, done, _ = env.step(action)
    i += 1
