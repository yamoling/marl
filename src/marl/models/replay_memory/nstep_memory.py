from rlenv import Transition
from ..batch import Batch
from .replay_memory import ReplayMemory
from .memory_wrapper import MemoryWrapper

class NStepReturnMemory(MemoryWrapper[Transition]):
    def __init__(self, wrapped: ReplayMemory[Transition], n: int, discout_factor: float):
        super().__init__(wrapped)
        # Gammas range from 1 to gamma ** (n-1) because the learning algorithm will also discount the reward
        self._gammas = [(e+1, discout_factor ** e) for e in range(n-1)]
        self._n  = n

    def __len__(self) -> int:
        return max(0, len(self.wrapped) - self._n)
    
    def get_batch(self, indices: list[int]) -> Batch:
        """The following operations are performed on the batch
        - replace the rewards by the discounted sum of the n-step returns
        - set the dones flags properly if any of the n transitions was done
        - set the obs_ to the n^th observation (or last of episode)
        """
        samples = []
        for index in indices:
            transition = self[index]
            obs = transition.obs
            action = transition.action
            reward = transition.reward
            done = transition.done
            obs_  = transition.obs_
            for i, gamma in self._gammas:
                current = self._memory[index + i]
                done = current.done
                reward += gamma * current.reward
                obs_ = current.obs_
                if done:
                    break
            samples.append(Transition(obs, action, reward, done, {}, obs_))
        return Batch.from_transitions(samples)
