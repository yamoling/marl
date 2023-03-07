from dataclasses import dataclass

@dataclass
class MemoryConfig:
    prioritized: bool
    size: int
    nstep: int

@dataclass
class TrainConfig:
    recurrent: bool
    logdir: str
    vdn: bool
    env_wrappers: list[str]
    time_limit: int
    level: str
    memory: MemoryConfig

@dataclass
class StartTrain:
    num_steps: int
    test_interval: int | None 
    num_tests: int | None

@dataclass
class StartTest:
    num_steps: int