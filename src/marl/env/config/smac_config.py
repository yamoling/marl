from dataclasses import KW_ONLY, dataclass

from marlenv.adapters import SMAC

from .env_config import EnvConfig


class _SMAC(SMAC):
    def step(self, action):
        """Represent SMAC's built-in episode limit as truncation. @ai-generated"""
        step = super().step(action)
        if step.info.get("episode_limit", False):
            step.done = False
            step.truncated = True
        return step


@dataclass
class SMACConfig(EnvConfig[_SMAC]):
    map_name: str
    _: KW_ONLY
    debug: bool = False
    game_version: str | None = None

    def make_base_env(self):
        return _SMAC(
            self.map_name,
            continuing_episode=True,
            debug=self.debug,
            game_version=self.game_version,
        )
