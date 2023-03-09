import rlenv
import marl
from laser_env import LevelGenerator, ObservationType, GeneratorWrapper



if __name__ == "__main__":
    logdir = "logs/test"
    generator = (LevelGenerator(10, 10, 2)
                .obs_type(ObservationType.LAYERED)
                .gems(3)
                .wall_density(0.15))
    
    env = (rlenv.Builder(GeneratorWrapper(generator, logdir))
           .agent_id()
           .time_limit(30)
           .build())
    qnetwork = marl.nn.model_bank.CNN.from_env(env)
    
    algo = (marl.DeepQBuilder()
            .vdn()
            .qnetwork(qnetwork)
            .build())
    algo = marl.wrappers.ReplayWrapper(algo, logdir)
    logger = marl.logging.TensorBoardLogger(logdir)
    runner = marl.Runner(env, algo, logger)
    runner.train(2_000_000)
    

