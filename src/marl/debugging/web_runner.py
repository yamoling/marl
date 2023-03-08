import marl
from marl.qlearning import IQLearning
from marl.models import ReplayMemory
from marl.logging import WSLogger



class WebRunner(marl.Runner):
    def __init__(self, env, test_env, algo: IQLearning, logdir: str, memory: ReplayMemory = None):
        super().__init__(env, test_env=test_env, algo=algo, logger=WSLogger(logdir))
        self.time_step = 0
        self.episode_num = 0
        self.obs = self._env.reset()
        self._algo.before_train_episode(self.episode_num)
        self.memory = memory
        self.stop = False
        # Type hinting
        self._algo: IQLearning
        self._logger: WSLogger
        # Start server
        self.write_experiment_summary()
        
    @property
    def port(self) -> int:
        return self._logger.port
