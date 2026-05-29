import pickle
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from typing import cast

from marlenv import MARLEnv

from .env_config import EnvConfig


@dataclass
class PickleEnvConfig(EnvConfig[MARLEnv]):
    pickle_path: str
    _: KW_ONLY

    def make_base_env(self):
        with open(self.pickle_path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def create[A](
        cls,
        env: MARLEnv[A],
        agent_id: bool = True,
        time_limit: int | None = None,
        last_action: bool = False,
        maven_noise_size: int | None = None,
        **kwargs,
    ) -> EnvConfig[MARLEnv[A]]:
        env_dir = Path("envs")
        env_dir.mkdir(exist_ok=True)
        env_file = env_dir / f"{env.name}.pkl"
        suffix = 0
        while env_file.exists():
            with open(env_file, "rb") as f:
                other = cast(MARLEnv[A], pickle.load(f))
            if other == env:
                # If we find a matching environment, just reuse the same pickle file.
                return cls(
                    pickle_path=f.name,
                    agent_id=agent_id,
                    time_limit=time_limit,
                    last_action=last_action,
                    maven_noise_size=maven_noise_size,
                    **kwargs,
                )
            suffix += 1
            env_file = env_dir / f"{env.name}-{suffix}.pkl"
        with open(env_file, "wb") as f:
            pickle.dump(env, f)
        return cls(
            pickle_path=f.name,
            agent_id=agent_id,
            time_limit=time_limit,
            last_action=last_action,
            maven_noise_size=maven_noise_size,
            **kwargs,
        )
