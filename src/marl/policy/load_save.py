from typing import Type
import inspect
from .policy import Policy


def _get_all_policies() -> dict[str, Type[Policy]]:
    from marl import policy
    classes = inspect.getmembers(policy, lambda x: inspect.isclass(x) and issubclass(x, Policy) and x != Policy)
    return {name: cls for name, cls in classes }

def from_summary(summary: dict[str, ]) -> Policy:
    return ALL_POLICIES[summary["name"]].from_summary(summary)

def register(policy: Type[Policy]):
    """Register a Policy"""
    ALL_POLICIES[policy.__name__] = policy

ALL_POLICIES: dict[str, Type[Policy]] = _get_all_policies()
