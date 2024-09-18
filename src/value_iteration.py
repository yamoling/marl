from itertools import product
from lle import World, WorldState, Action, EventType
from abc import abstractmethod, ABC
from datetime import datetime


class MDP[S, A](ABC):
    """Adversarial Markov Decision Process"""

    @abstractmethod
    def is_final(self, state: S) -> bool:
        """Returns true if the given state is final (i.e. the game is over)."""

    @abstractmethod
    def available_actions(self, state: S) -> list[A]:
        """Returns the list of available actions for the current agent from the given state."""

    @abstractmethod
    def transitions(self, state: S, action: A) -> list[tuple[S, float]]:
        """Returns the list of next states with the probability of reaching it by performing the given action."""

    @abstractmethod
    def states(self) -> list[S]:
        """Returns the list of all states."""

    @abstractmethod
    def reward(self, state: S, action: A, new_state) -> float:
        """Reward function"""


class WorldMDP(MDP[WorldState, list[Action]]):
    def __init__(self, world: World):
        super().__init__()
        self.world = world

    def available_actions(self, state: WorldState):
        self.world.set_state(state)
        available = self.world.available_actions()
        return list(product(*available))

    def transitions(self, state: WorldState, action: list[Action]) -> list[tuple[WorldState, float]]:
        self.world.set_state(state)
        self.world.step(action)
        return [(self.world.get_state(), 1.0)]

    def is_final(self, state: WorldState) -> bool:
        self.world.set_state(state)
        if any(agent.is_dead for agent in self.world.agents):
            return True
        if all(agent.has_arrived for agent in self.world.agents):
            return True
        return False

    def reward(
        self,
        state: WorldState,
        action: list[Action],
        new_state: WorldState,
    ) -> float:
        # Step the world and check if the new state is the same as the given one
        # If if is not the same, then test all the other available actions.
        self.world.set_state(state)
        events = self.world.step(action)
        actual_new_state = self.world.get_state()
        if actual_new_state != new_state:
            raise ValueError("The new state is not reachable from the given state")
        reward = 0.0
        for e in events:
            match e.event_type:
                case EventType.GEM_COLLECTED:
                    reward += 1.0
                case EventType.AGENT_DIED:
                    reward -= 1.0
                case EventType.AGENT_EXIT:
                    reward += 1.0
        if all(agent.has_arrived for agent in self.world.agents):
            reward += 1.0
        return reward

    def states(self):
        all_positions = set(product(range(self.world.height), range(self.world.width)))
        all_positions = all_positions.difference(set(self.world.wall_pos))
        agents_positions = product(all_positions, repeat=self.world.n_agents)
        collection_status = product([True, False], repeat=self.world.n_gems)

        for pos, collected in product(agents_positions, collection_status):
            # We must prevent states where two agents are on the same tile
            pos_0 = pos[0]
            if any(pos_0 == p for p in pos[1:]):
                print(f"Skipping state {pos}")
                continue
            s = WorldState(list(pos), list(collected))
            yield s


class ValueIteration[S, A]:
    def __init__(self, mdp: MDP[S, A], gamma: float):
        self.mdp = mdp
        self.gamma = gamma
        self.values = dict[S, float]()

    def value(self, state: S) -> float:
        """Returns the value of the given state."""
        return self.values.get(state, 0.0)

    def policy(self, state: S):
        """Returns the action that maximizes the Q-value of the given state."""
        max_action = None
        max_value = -float("inf")
        for action in self.mdp.available_actions(state):
            value = self.qvalue(state, action)
            if value > max_value:
                max_value = value
                max_action = action
        return max_action

    def qvalue(self, state: S, action: A) -> float:
        """Returns the Q-value of the given state-action pair based on the state values."""
        qvalue = 0.0
        for next_state, prob in self.mdp.transitions(state, action):
            reward = self.mdp.reward(state, action, next_state)
            qvalue += prob * (reward + self.gamma * self.value(next_state))
        return qvalue

    def _compute_value_from_qvalues(self, state: S) -> float:
        """
        Returns the value of the given state based on the Q-values.

        This is a private method, meant to be used by the value_iteration method.
        """
        max_value = -float("inf")
        for action in self.mdp.available_actions(state):
            value = self.qvalue(state, action)
            max_value = max(max_value, value)
        return max_value

    def value_iteration(self, n: int):
        start = datetime.now()
        import os
        import pickle

        directory = "logs-value-iteration"
        os.makedirs(directory, exist_ok=True)
        filename = f"{directory}/logs-{start.strftime("%Y-%d-%m_%H-%M-%S")}.txt"
        with open(filename, "a") as log:
            log.write(f"Started at {start}\n")
        for i in range(n):
            print(f"Iteration {i}")
            values = {}
            for state in self.mdp.states():
                if self.mdp.is_final(state):
                    values[state] = 0.0
                else:
                    values[state] = self._compute_value_from_qvalues(state)
            self.values = values
            with open(filename, "a") as log:
                duration = datetime.now() - start
                log.write(f"Iteration: {i}, duration: {duration}\n")
            print("Saving values")
            with open(f"{directory}/values-{i}.pkl", "wb") as p:
                pickle.dump(self.values, p)
        with open(filename, "a") as log:
            log.write(f"Finished at {datetime.now()}\n")


if __name__ == "__main__":
    print("Value Iteration")
    mdp = WorldMDP(World.level(6))
    algo = ValueIteration(mdp, gamma=0.95)
    algo.value_iteration(n=100)
