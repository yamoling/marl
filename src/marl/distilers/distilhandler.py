import os
import pathlib

from typing import Literal, Optional

import numpy as np

from marl.distilers.sdt import SoftDecisionTree
from marl.models import Runner, Experiment
from marl.agents.qlearning import DQN
from marl.utils.gpu import get_device

class DistilHandler:
    _distiler: SoftDecisionTree
    _experiment: Experiment
    _agent: DQN

    def __init__(self,
                 experiment: Experiment,
                 distiler: SoftDecisionTree            
    ):
        self._experiment = experiment
        self._distiler = experiment
        self._agent = experiment.agent
        if self._agent.last_qvalues == None:
            self._agent.last_qvalues = np.ndarray(0)


    @staticmethod
    def create(logdir: str):
        # check if there has been a run

        experiment = Experiment.load(logdir)
        if experiment.agent.name == "DQN":
            if experiment.env.reward_space.size == 1:
                output_shape = (experiment.env.n_actions,)
            else:
                output_shape = (experiment.env.n_actions, experiment.env.reward_space.size)
            distiler = SoftDecisionTree(
                input_shape = experiment.env.observation_shape,
                output_shape = output_shape,


            )
            distil_handler = DistilHandler(experiment, distiler)
            return distil_handler
        # Get runner, do perform_one_test
        # makes one episode as test (on trained agent)
        else:
            raise NotImplementedError(f"Distilation not implemented for agent {experiment.agent.name}")
    
    def run(self, 
            #seed: int =0,
            fill_strategy: Literal["scatter", "group"] = "scatter",
            required_memory_MB: int = 0,
            quiet: bool = False,
            device: Literal["cpu", "auto"] | int = "auto",
            #n_episodes: int,
            ):
        # Select run, for now last run
        runs = os.listdir(self._experiment.logdir)
        runs = [run for run in runs if "run" in run]
        run = runs[-1]
        run_path = os.path.join(self._experiment.logdir,run)
        tests_path = os.path.join(run_path,"test")
        tests = os.listdir(tests_path)
        tests.sort(key=int)
        test = tests[-1]
        load_test_path = os.path.join(tests_path,test)
        self._agent.load(load_test_path)
        
        runner = Runner(
            env = self._experiment.env,
            agent = self._agent,
            log_qvalues = True) # Need to keep for action distribution
        selected_device = get_device(device, fill_strategy, required_memory_MB)
        runner = runner.to(selected_device)

        episode = runner.perform_one_test()

        obs = episode.all_observations
        qv = episode.qvalues


