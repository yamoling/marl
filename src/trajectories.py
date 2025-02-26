import orjson
import os
import marl

# env = LLE.level(6).build()
exp = marl.Experiment.load("logs/LLE-lvl6-PBRS-VDN")
env = exp.env
print(env.observation_shape)
print(env.extra_shape)
print(env.extras_meanings)
root = "logs/LLE-lvl6-PBRS-VDN/run_2025-02-24_17:29:11.134180_seed=0/test"
for item in os.listdir(root):
    print(item)
    if not item.isnumeric():
        continue
    with open(os.path.join(root, item, "0", "actions.json"), "r") as f:
        actions = orjson.loads(f.read())
    seed = int(item)
    episode = env.replay(actions, seed)
    imgs = episode.get_images(env, seed)
