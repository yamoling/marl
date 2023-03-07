import rlenv
import marl
from laser_env import LaserEnv


if __name__ == "__main__":
    exp = marl.models.Experiment("logs/maps_lvl3")
    print(exp.from_checkpoint(""))
