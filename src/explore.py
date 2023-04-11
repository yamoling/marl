import marl
import time



if __name__ == '__main__':
    experiment = marl.Experiment.load("logs/dynamic-vdn-laser-softmax")
    episode = experiment.replay_episode("logs/dynamic-vdn-laser-softmax/run_1680809609.2831898/test/0/3")
    
