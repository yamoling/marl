import os

from typing import Literal, Optional

import numpy as np

from marl.distilers.sdt import SoftDecisionTree
from marl.models import Experiment
from marl.agents.qlearning import DQN
from marl.utils.gpu import get_device

class DistilHandler:
    _distiler: SoftDecisionTree # or sklearn DT/Randomforest?
    _experiment: Experiment
    _agent: DQN

    def __init__(self,
                 experiment: Experiment,
                 distiler: SoftDecisionTree # or sklearn DT/Randomforest?           
    ):
        self._experiment = experiment
        self._distiler = distiler
        self._agent = experiment.agent
        if self._agent.last_qvalues == None:
            self._agent.last_qvalues = np.ndarray(0)


    @staticmethod
    def create(logdir: str, exp_dataset: bool, dataset: str):
        # check if there has been a run

        experiment = Experiment.load(logdir)
        if experiment.agent.name == "DQN": # Note: Other similar agents should work, but for our use case we'll limit to DQN and to specific transformations to the output
            #if experiment.env.reward_space.size == 1:
            if dataset == "action":
                raise NotImplementedError(f"Distilation not implemented for single action output yet.")
            else:
                output_shape = (experiment.env.n_agents, experiment.env.n_actions,) # for now consider output for all agents and see, might have to do one distilled model per agent
            #else:
            #    output_shape = (experiment.env.n_actions, experiment.env.reward_space.size)
            # Force to not be MO?
            # Distiler may also be other models, consider simple DT and RandomForest later
            distiler = SoftDecisionTree( # or sklearn DT/Randomforest?
                input_shape = experiment.env.observation_shape, # Can't consider extras, because we focus on the observation
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
        # Move to prepare_datset, observation "flattening" on tower BX?
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
        
        
        selected_device = get_device(device, fill_strategy, required_memory_MB)
        self._agent.to(selected_device)

        outputs, inputs = self.perform_one_test()

        assert inputs[0][0].shape == self._distiler.input_shape and outputs[0].shape == self._distiler.output_shape 
        # Once dataset is ready we train the model, save it too
        # Later find a way to simply use the model given an observation or experiment
            # In those cases run both models and note the faithfulness of the distilled model.
            # For experimentation: possibility to compare multiple distillations
        
    def prepare_dataset():
        """
        Prepares the dataset used to train a distilled model, depending on the type given as argument and whether it's to be extended or not.
        """
        pass

    def perform_one_test(self, seed: Optional[int] = None):
            """
            Perform a single test episode.

            The test can be seeded for reproducibility purposes, for instance when the policy or the environment is stochastic.
            """
            self._agent.set_testing()
            if seed is not None:
                self._experiment.env.seed(seed)
                self._agent.seed(seed)
            self._agent.new_episode()

            distributions = []
            observations = []
            i = 0
            is_finished = False
            while not is_finished:
                if i == 0: obs, state = self._experiment.env.reset()
                else:
                    obs = step.obs
                i += 1
                distr = self._agent.get_action_distribution(obs)
                step = self._experiment.env.step(self._agent.policy.get_action(distr, obs.available_actions))
                distributions.append(distr)
                observations.append(obs.data)
                is_finished = step.done | step.truncated
            return np.array(distributions), np.array(observations)