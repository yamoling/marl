import logging
from dataclasses import KW_ONLY, dataclass

from marlenv.adapters import SMAC

from .env_config import EnvConfig

logger = logging.getLogger(__name__)


class _SMAC(SMAC):
    def step(self, action):
        """
        Represent SMAC's built-in episode limit as truncation, and keep the terminal metric schema stable.

        SMAC only puts `episode_limit` in `info` when the limit is reached, and `Episode` promotes every
        boolean of the terminal `info` into a metric. Leaving the flag in place would therefore give
        truncated and terminated episodes different metric schemas, which breaks the test aggregation and
        forces the CSV logger to rewrite its files. The flag is consumed here instead, since `truncated`
        already carries the same information.

        Separately, `StarCraft2Env.step` returns `(0, True, {})` (an empty `info`) when the SC2 binary
        raises a `protocol.ProtocolError`/`ConnectionError` and it has to `full_restart()` the game. That
        transient engine hiccup would otherwise omit `battle_won` from one terminal step out of thousands,
        which crashes `simple_runner._test_and_log`'s cross-episode aggregation and takes an entire
        multi-hour run down with it (observed after ~1-14h of training on this repository's SMAC runs).
        `battle_won` defaults to False here, matching SMAC's own default before a win is confirmed.

        @ai-generated
        """
        step = super().step(action)
        if step.info.pop("episode_limit", False):
            step.done = False
            step.truncated = True
        if (step.done or step.truncated) and "battle_won" not in step.info:
            logger.warning(
                "SMAC terminal step is missing 'battle_won' (likely a full_restart after a protocol error); defaulting to False."
            )
            step.info["battle_won"] = False
        return step


@dataclass
class SMACConfig(EnvConfig[_SMAC]):
    """
    Serializable configuration of a SMAC scenario.

    The reward-related attributes mirror the `StarCraft2Env` constructor. The sparse-reward setting of the
    LAIES paper is `reward_sparse=True` together with `reward_scale=False`, which yields exactly +1 for a
    win, -1 for a defeat and 0 otherwise (SMAC divides even sparse rewards by `max_reward / reward_scale_rate`
    when `reward_scale` is left enabled).
    """

    map_name: str
    _: KW_ONLY
    debug: bool = False
    game_version: str | None = None
    difficulty: str = "7"
    step_mul: int = 8
    reward_sparse: bool = False
    reward_scale: bool = True
    reward_scale_rate: int = 20
    reward_only_positive: bool = True
    reward_death_value: int = 10
    reward_win: int = 200
    reward_defeat: int = 0
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
