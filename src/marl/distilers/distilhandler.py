import os

from typing import Literal, Optional
from random import sample

import numpy as np

from marl.distilers.sdt import SoftDecisionTree
from marl.models import Experiment
from marl.agents.qlearning import DQN
from marl.utils.gpu import get_device

class DistilHandler:
    _distiler: SoftDecisionTree # or sklearn DT/Randomforest?
    _experiment: Experiment
    _agent: DQN

    n_agents: int

    def __init__(self,
                 experiment: Experiment,
                 distiler: SoftDecisionTree # or sklearn DT/Randomforest?           
    ):
        self._experiment = experiment
        self._distiler = distiler
        self._agent = experiment.agent
        if self._agent.last_qvalues == None:
            self._agent.last_qvalues = np.ndarray(0)

        self.n_agents = self._experiment.env.n_agents


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
            #   output_shape = (experiment.env.n_agents, experiment.env.n_actions,) # for now consider output for all agents and see
            #else:
            #    output_shape = (experiment.env.n_actions, experiment.env.reward_space.size)
            # Force to not be MO?
            # Distiler may also be other models, consider simple DT and RandomForest later
            distiler = SoftDecisionTree(
                input_shape = experiment.env.observation_shape[1]*experiment.env.observation_shape[2], # Can't consider extras, because we focus on the observation - ALSO SPECIFIC FOR LAYERED OBS
                output_shape = experiment.env.n_actions,
                max_depth=5
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

        outputs, inputs = self.simulate_one_episode()

        assert inputs.shape[-1] == self._distiler.input_shape and outputs.shape[-1] == self._distiler.output_shape 

        #episode_len = len(distributions)//self.n_agents
        #for i in range (self.n_agents):
        #    sl = slice(i*episode_len,(i+1)*episode_len)
        observations = np.transpose(inputs,(1,0,2))
        distributions = np.transpose(outputs,(1,0,2))
        for i in range(10):
            self._distiler.train_(inputs,outputs, i)
        
        self._distiler.test_(inputs,outputs, i)
       
        #self._distiler.test_(
        #    np.random.choice(observations.reshape(-1,self._distiler.input_shape),self._distiler.batch_size),
        #    np.random.choice(distributions.reshape(-1,self._distiler.output_shape),self._distiler.batch_size))
        
    def prepare_dataset():
        """
        Prepares the dataset used to train a distilled model, depending on the type given as argument and whether it's to be extended or not.
        """
        pass

    def simulate_one_episode(self, seed: Optional[int] = None):
            """
            Runs an episode to gather action distributions and observations.
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
                else: obs = step.obs
                i += 1
                distr, act = self._agent.get_action_distribution(obs)
                step = self._experiment.env.step(act)
                
                # This part is very specific to my current use case: to adapt and make more generic!!
                temp = obs.data
                #distributions.extend(distr)
                #observations.extend(self.flatten_observation(temp, axis=1).reshape(self.n_agents,-1)) # axis one because axis 0 is agent
                
                distributions.append(distr)
                observations.append(self.flatten_observation(temp, axis=1).reshape(self.n_agents,-1)) # axis one because axis 0 is agent
                
                is_finished = step.done | step.truncated
            return np.array(distributions), np.array(observations)
    
    def flatten_observation(self, observation, axis=0):
        observation = np.array(observation)
        flattened_obs = np.full((self.n_agents, observation.shape[axis+1], observation.shape[axis+2]), -1, dtype=int)

        # Find the first n (axis 0) where O[n, i, j] == 1
        # This gives a mask of the same shape as O
        mask = observation == 1

        # Get the first 'n' where the condition is met along axis 0
        first_n = np.argmax(mask, axis=axis)+1

        # Check if *any* 1 was found along axis 0 for each (i, j)
        any_valid = mask.any(axis=axis)

        # Only update F where a 1 was found
        flattened_obs[any_valid] = first_n[any_valid]

        # Identify agent self-layer: when layer == agent index
        # Extract self-layer per agent: O[a, a] for all a → shape (4, 12, 13)
        agent_indices = np.arange(self.n_agents)
        agent_self_mask = observation[agent_indices, agent_indices] == 1

        # Set agent's own positions to 0
        flattened_obs[agent_self_mask] = 0

        return flattened_obs