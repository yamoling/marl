from rlenv import Episode, Transition, Observation
from marl import RLAlgo


class AlgorithmWrapper(RLAlgo):
    def __init__(self, algo: RLAlgo) -> None:
        super().__init__()
        self.algo = algo

    def choose_action(self, observation: Observation):
        return self.algo.choose_action(observation)

    def summary(self) -> dict:
        return self.algo.summary()

    def save(self, to_path: str):
        return self.algo.save(to_path)

    def load(self, from_path: str):
        return self.algo.load(from_path)

    def before_tests(self):
        return self.algo.before_tests()

    def after_tests(self, episodes: list[Episode], time_step: int):
        return self.algo.after_tests(episodes, time_step)

    def after_step(self, transition: Transition, step_num: int):
        return self.algo.after_step(transition, step_num)
    
    def before_episode(self, episode_num: int):
        return self.algo.before_episode(episode_num)
    
    def after_episode(self, episode_num: int, episode: Episode):
        return self.algo.after_episode(episode_num, episode)
