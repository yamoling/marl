from dataclasses import dataclass

@dataclass
class MemoryConfig:
    prioritized: bool
    size: int
    nstep: int

@dataclass
class GeneratorConfig:
    n_gems: int
    n_agents: int
    n_lasers: int
    wall_density: float
    width: int
    height: int

@dataclass
class ExperimentConfig:
    recurrent: bool
    logdir: str
    vdn: bool
    env_wrappers: list[str]
    time_limit: int
    time_penalty: float
    forced_actions: dict[int, int]
    level: str
    obs_type: str
    static_map: bool
    memory: MemoryConfig
    generator: GeneratorConfig

@dataclass
class TrainConfig:
    """The logdir is the key from which to know what experiment to train"""
    num_steps: int
    test_interval: int | None 
    num_tests: int | None
