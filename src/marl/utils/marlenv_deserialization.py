"""Decode marlenv dataclasses serialized by orjson.

JSON does not retain NumPy dtype metadata. Typed fields use marlenv's
float32/bool conventions; actions infer their dtype from their values.
Untyped metadata (info, other, metrics) is left as decoded by JSON.
"""

from typing import Any

import numpy as np
from marlenv import Episode, Observation, State, Transition


def observation_from_dict(d: dict[str, Any]) -> Observation:
    """Restore arrays and let the constructor recompute n_agents. @ai-generated"""
    return Observation(
        data=np.asarray(d["data"], dtype=np.float32),
        available_actions=np.asarray(d["available_actions"], dtype=np.bool_),
        extras=np.asarray(d["extras"], dtype=np.float32),
    )


def state_from_dict(d: dict[str, Any]) -> State:
    """Restore array states while retaining scalar/generic state data. @ai-generated"""
    data = d["data"]
    if isinstance(data, (list, np.ndarray)):
        data = np.asarray(data, dtype=np.float32)
    return State(data=data, extras=np.asarray(d["extras"], dtype=np.float32))


def transition_from_dict(d: dict[str, Any]) -> Transition:
    """Restore nested objects and extra values without nesting other. @ai-generated"""
    transition = Transition(
        obs=observation_from_dict(d["obs"]),
        state=state_from_dict(d["state"]),
        action=np.asarray(d["action"]),
        reward=np.asarray(d["reward"], dtype=np.float32),
        done=d["done"],
        info=d["info"],
        next_obs=observation_from_dict(d["next_obs"]),
        next_state=state_from_dict(d["next_state"]),
        truncated=d["truncated"],
    )
    transition.other = dict(d["other"])
    return transition


def episode_from_dict(d: dict[str, Any]) -> Episode:
    """Restore the per-step arrays without rebuilding episode metadata. @ai-generated"""
    for key in (
        "all_observations",
        "all_extras",
        "rewards",
        "all_states",
        "all_states_extras",
    ):
        d[key] = [np.asarray(value, dtype=np.float32) for value in d[key]]
    d["all_available_actions"] = [np.asarray(value, dtype=np.bool_) for value in d["all_available_actions"]]
    d["actions"] = [np.asarray(value) for value in d["actions"]]
    return Episode(**d)
