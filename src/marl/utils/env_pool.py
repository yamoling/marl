import rlenv
from rlenv import RLEnv
from rlenv.models import RLEnv
from laser_env import Difficulty, ObservationType
import random

from rlenv.wrappers import RLEnvWrapper

class EnvPool(RLEnvWrapper):
    def __init__(self, envs: list[RLEnv]):
        super().__init__(envs[0])
        self._pool = envs

    def reset(self):
        self.wrapped = random.choice(self._pool)
        return super().reset()

    def kwargs(self) -> dict[str, str]:
        return {
            "envs": [env.summary() for env in self._pool]
        }
    
    @classmethod
    def from_summary(cls, env: RLEnv, summary: dict[str,]) -> "RLEnvWrapper":
        kwargs = summary.pop(cls.__name__)
        envs = [rlenv.from_summary(summary) for summary in kwargs["envs"]]
        return cls(envs)

    def seed(self, seed_value: int):
        for env in self._pool:
            env.seed(seed_value)

# class EnvPool(RLEnv):
#     def __init__(self, envs: list[RLEnv]):
#         self._pool = envs
#         self.env = random.choice(self._pool)
#         super().__init__(self.env.action_space)
        

#     @property
#     def action_space(self):
#         return self.env.action_space
    
#     @property
#     def observation_shape(self):
#         return self.env.observation_shape
    
#     @property
#     def state_shape(self) -> tuple[int, ...]:
#         return self.env.state_shape
    
#     def reset(self):
#         self.env = random.choice(self._pool)
#         return self.env.reset()
    
#     def step(self, action):
#         return self.env.step(action)
    
#     def render(self, mode="human"):
#         return self.env.render(mode)

#     def kwargs(self) -> dict[str, str]:
#         return {
#             "envs": [env.summary() for env in self._pool]
#         }
    
#     @classmethod
#     def from_summary(cls, summary: dict[str,]) -> RLEnv:
#         kwargs = summary.pop(cls.__name__)
#         envs = [rlenv.from_summary(summary) for summary in kwargs["envs"]]
#         env = cls(envs)
#         current_env = summary.pop("current_env_index")
#         env.env = envs[current_env]
#         return env

#     def summary(self) -> dict[str, ]:
#         return {
#             **super().summary(),
#             "current_env_index": self._pool.index(self.env)
#         }
    
#     def get_state(self):
#         return self.env.get_state()
    
#     def get_avail_actions(self):
#         return self.env.get_avail_actions() 
    
#     def seed(self, seed_value: int):
#         for env in self._pool:
#             env.seed(seed_value)
    

   
rlenv.register_wrapper(EnvPool)


def extract_envs(zip_path: str, difficulty: Difficulty, obs_type: ObservationType) -> list[RLEnv]:
    import os
    import zipfile
    import tempfile
    import laser_env
    import shutil
    train_envs: list[laser_env.LaserEnv] = []
    test_envs: list[laser_env.LaserEnv] = []
    tmp_dir = tempfile.mkdtemp()
    os.makedirs(tmp_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zip_file:
        zip_file.extractall(tmp_dir)
    train_dir = os.path.join(tmp_dir, os.path.basename(zip_path)[:-4], difficulty.name, "train")
    test_dir = os.path.join(tmp_dir, os.path.basename(zip_path)[:-4], difficulty.name, "test")
    for file in os.listdir(train_dir):
        if not file.endswith(".txt"):
            continue
        file = os.path.join(train_dir, file)
        train_envs.append(laser_env.StaticLaserEnv(file, obs_type))
    for file in os.listdir(test_dir):
        if not file.endswith(".txt"):
            continue
        file = os.path.join(test_dir, file)
        test_envs.append(laser_env.StaticLaserEnv(file, obs_type))
    shutil.rmtree(tmp_dir)
    return train_envs, test_envs


def pool_from_zip(zip_path: str, difficulty: Difficulty | int, obs_type: ObservationType) -> tuple[EnvPool, EnvPool]:
    train_envs, test_envs = extract_envs(zip_path, difficulty, obs_type)
    return EnvPool(train_envs), EnvPool(test_envs)