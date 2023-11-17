from rlenv import Observation
from marl import RLAlgo


class AlgorithmWrapper(RLAlgo):
    def __init__(self, algo: RLAlgo) -> None:
        super().__init__()
        self.algo = algo

    def choose_action(self, observation: Observation):
        return self.algo.choose_action(observation)

    def save(self, to_path: str):
        return self.algo.save(to_path)

    def load(self, from_path: str):
        return self.algo.load(from_path)
