import rlenv
from rlenv import RLEnv
from rlenv.wrappers import RLEnvWrapper
import random

class EnvPool(RLEnvWrapper):
    def __init__(self, envs: list[RLEnv]|list[str]):
        if isinstance(envs[0], str):
            import rlenv
            envs = [rlenv.from_summary(env_summary) for env_summary in envs]
        super().__init__(envs[0])
        self.envs = envs
        self.env = envs[0]

    def reset(self):
        self.env = random.choice(self.envs)
        return super().reset()

    def kwargs(self) -> dict[str, str]:
        return {
            "envs": [env.summary() for env in self.envs]
        }
    
    @classmethod
    def from_summary(cls, env: RLEnv, summary: dict[str,]) -> "RLEnvWrapper":
        kwargs = summary.pop(cls.__name__)
        envs = [rlenv.from_summary(summary) for summary in kwargs["envs"]]
        return cls(envs)
