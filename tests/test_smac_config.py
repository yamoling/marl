"""Exercise SMACConfig without starting StarCraft II."""

from unittest.mock import MagicMock

import numpy as np

from marl.env import SMACConfig


def test_smac_config_round_trip_preserves_adapter_options(monkeypatch):
    created = []

    class FakeEngine:
        def __init__(self, **kwargs):
            self.n_agents = 2
            self.n_actions = 3
            self.map_name = kwargs["map_name"]
            self._sc2_proc = None
            self.get_env_info = MagicMock(return_value={"obs_shape": 4, "state_shape": 8})
            self.get_obs = MagicMock(return_value=np.zeros((2, 4), dtype=np.float32))
            self.get_state = MagicMock(return_value=np.zeros(8, dtype=np.float32))
            self.get_avail_actions = MagicMock(return_value=np.ones((2, 3), dtype=np.int32))
            self.seed = MagicMock(return_value=kwargs.get("seed", 0))
            self.reset = MagicMock(return_value=(self.get_obs(), self.get_state()))
            self.step = MagicMock(return_value=(1.0, True, {"episode_limit": True}))
            self.close = MagicMock()
            created.append((kwargs, self))

    monkeypatch.setattr("marlenv.adapters.smac_adapter.StarCraft2Env", FakeEngine)
    config = SMACConfig("corridor", game_version="4.6.2", debug=True, last_action=True)
    restored = SMACConfig.from_json(config.to_json())
    env = restored.make()
    env.seed(123)
    obs, _ = env.reset()
    assert obs.extras.shape == (2, 5)  # agent IDs and previous actions
    assert created[-1][0]["game_version"] == "4.6.2"
    assert created[-1][0]["debug"] is True
    assert created[-1][0]["continuing_episode"] is True
    assert created[-1][0]["seed"] == 123
    created[-2][1].close.assert_called_once()
    step = env.step(np.zeros(2, dtype=np.int64))
    assert step.truncated and not step.done
    created[-1][1].step.return_value = (1.0, True, {"battle_won": True})
    step = env.step(np.zeros(2, dtype=np.int64))
    assert step.done and not step.truncated
    created[-1][1].close()
