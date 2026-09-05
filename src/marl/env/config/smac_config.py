from dataclasses import KW_ONLY, dataclass

from marlenv.adapters import SMAC

from .env_config import EnvConfig


class _SMAC(SMAC):
    def step(self, action):
        """
        Represent SMAC's built-in episode limit as truncation.

        SMAC only puts `episode_limit` in `info` when the limit is reached, and `Episode` promotes every
        boolean of the terminal `info` into a metric. Leaving the flag in place would therefore give
        truncated and terminated episodes different metric schemas, which breaks the test aggregation and
        forces the CSV logger to rewrite its files. The flag is consumed here instead, since `truncated`
        already carries the same information.

        @ai-generated
        """
        step = super().step(action)
        if step.info.pop("episode_limit", False):
            step.done = False
            step.truncated = True
        return step


@dataclass
class SMACConfig(EnvConfig[_SMAC]):
    """
    Serializable configuration of a SMAC scenario.

    The reward-related attributes mirror the `StarCraft2Env` constructor. The sparse-reward setting of the
    LAIES paper is `reward_sparse=True` together with `reward_scale=False`, which yields exactly +1 for a
    win, -1 for a defeat and 0 otherwise (SMAC divides even sparse rewards by `max_reward / reward_scale_rate`
    when `reward_scale` is left enabled).

    @ai-generated
    """

    map_name: str
    _: KW_ONLY
    debug: bool = False
    game_version: str | None = None
    difficulty: str = "7"
    step_mul: int = 8
    reward_sparse: bool = False
    reward_scale: bool = True
    reward_scale_rate: float = 20
    reward_only_positive: bool = True
    reward_death_value: float = 10
    reward_win: float = 200
    reward_defeat: float = 0
    reward_negative_scale: float = 0.5

    def make_base_env(self):
        return _SMAC(
            self.map_name,
            continuing_episode=True,
            debug=self.debug,
            game_version=self.game_version,
            difficulty=self.difficulty,
            step_mul=self.step_mul,
            reward_sparse=self.reward_sparse,
            reward_scale=self.reward_scale,
            reward_scale_rate=self.reward_scale_rate,
            reward_only_positive=self.reward_only_positive,
            reward_death_value=self.reward_death_value,
            reward_win=self.reward_win,
            reward_defeat=self.reward_defeat,
            reward_negative_scale=self.reward_negative_scale,
        )
