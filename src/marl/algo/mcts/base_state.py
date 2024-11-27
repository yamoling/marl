from abc import ABC, abstractmethod
from typing import Self


class BaseAction(ABC):
    @abstractmethod
    def __eq__(self, other): ...

    @abstractmethod
    def __hash__(self): ...


class BaseState[A: BaseAction](ABC):
    """
    Baseclass for all states of a Monte Carlo Tree Search.

    This describes the state of the game/world, and the actions that can be taken from it.
    """

    @abstractmethod
    def get_possible_actions(self) -> list[A]:
        """
        Returns a list of all possible actions that can be taken from this state.

        Returns
        -------
        [A]: a list of all possible actions that can be taken from this state
        """

    @abstractmethod
    def take_action(self, action: A) -> tuple[Self, float]:
        """
        Returns the (state, reward) that results from taking the given action.

        Parameters
        ----------
        action: [any] BaseAction the action to take

        Returns
        -------
        BaseState: the state that results from taking the given action
        float: the reward that results from taking the given action
        """

    @property
    @abstractmethod
    def is_terminal(self) -> bool:
        """
        Returns whether this state is a terminal state.

        Returns
        -------
        bool: whether this state is a terminal state
        """
