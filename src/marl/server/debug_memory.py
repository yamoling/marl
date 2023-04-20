import numpy as np
from marl.models import ReplayMemory, PrioritizedMemory

class DebugMemory(ReplayMemory):
    def __init__(self, wrapped: ReplayMemory):
        super().__init__(wrapped.max_size)
        self.wrapped = wrapped

    def get_priorities(self):
        match self.wrapped:
            case PrioritizedMemory() as pm: 
                p = []
                for i in range(len(pm._tree)):
                    p.append(pm._tree[i])
                return p
            case _:
                return np.ones(len(self.wrapped), dtype=np.float32)

    def add(self, item):
        self.wrapped.add(item)

    def sample(self, batch_size):
        return self.wrapped.sample(batch_size)
    
    def update(self, indices, qvalues, qtargets):
        self.wrapped.update(indices, qvalues, qtargets)

    def get_batch(self, indices):
        return self.wrapped.get_batch(indices)
    
    def __len__(self):
        return len(self.wrapped)
    
    