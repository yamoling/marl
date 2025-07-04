import os

from typing import Literal, Optional
from random import sample

import numpy as np

from marl.xmarl.distilers.sdt import SoftDecisionTree
from marl.xmarl.distilers.utils import flatten_observation
from marl.models import Experiment
from marl.agents.qlearning import DQN
from marl.utils.gpu import get_device

class DistilHandler:
    """
    Class handling a distillation process for various distiller types and varying in-/outputs.
    Based on parameters prepares the dataset, loads the original model and creates the distilled model. Trains, tests and saves the final distilled model.
    """
    _distilers: list[SoftDecisionTree] # or sklearn DT/Randomforest?
    _experiment: Experiment
    _agent: DQN

    n_agents: int
    extras: bool
    individual_agents: bool

    def __init__(self,
                 experiment: Experiment,
                 distilers: list[SoftDecisionTree], # or sklearn DT/Randomforest?  
                 extras: bool,
                 individual_agents: bool = False, 
    ):
        self._experiment = experiment
        self._distilers = distilers
        self._agent = experiment.agent
        if self._agent.last_qvalues is None:
            self._agent.last_qvalues = np.ndarray(0)

        self.n_agents = self._experiment.env.n_agents
        self.individual_agents = individual_agents
        self.extras = extras


    @staticmethod
    def create(logdir: str, exp_dataset: bool, distiler: str, input_type: str, output_type: str, extras: bool, individual_agents: bool):
        # check if there has been a run

        experiment = Experiment.load(logdir)
        # Some weird things with how logdir is handeled by experiment
        if experiment.logdir != logdir:
            experiment.logdir = logdir
        if experiment.agent.name == "DQN": # Note: Other similar agents should work, but for our use case we'll limit to DQN and to specific transformations to the output
            #if experiment.env.reward_space.size == 1:
            if distiler != "sdt":
                raise NotImplementedError(f"Distilation to other model than SDT not implemented yet.")
            elif input_type == "abstracted_obs":
                raise NotImplementedError(f"Distilation not implemented for abstracted observation yet.")
            elif output_type == "action":
                raise NotImplementedError(f"Distilation not implemented for single action output yet.")
            # Force to not be MO?
            if extras: input_shape = experiment.env.observation_shape[1]*experiment.env.observation_shape[2]+experiment.env.extras_shape[0]+2
            else: input_shape = experiment.env.observation_shape[1]*experiment.env.observation_shape[2]
            # Distiler may also be other models, consider simple DT and RandomForest later
            distilers = []
            if individual_agents:
                for i in range (experiment.env.n_agents):
                    distil_model = SoftDecisionTree(
                        input_shape = input_shape,
                        output_shape = experiment.env.n_actions,
                        logdir = experiment.logdir,
                        max_depth=4,
                        bs=1,
                        n_agent=experiment.env.n_agents,
                        agent_id=i
                    )
                    distilers.append(distil_model)
            else:
                distil_model = SoftDecisionTree(
                    input_shape = input_shape,
                    output_shape = experiment.env.n_actions,
                    logdir = experiment.logdir,
                    max_depth=4,
                    bs=experiment.env.n_agents,
                    n_agent=experiment.env.n_agents,
                )
                distilers.append(distil_model)
            distil_handler = DistilHandler(experiment, distilers, extras, individual_agents)
            return distil_handler
        # Get runner, do perform_one_test
        # makes one episode as test (on trained agent)
        else:
            raise NotImplementedError(f"Distilation not implemented for agent {experiment.agent.name}")
            # Modify to check if qvalues of experiment is true or not, if true we can access them if not no, also only do it if getting qvalues, else distribution/action should always be accessible, albeit by bypassing something
    
    def run(self, 
            #seed: int =0,
            fill_strategy: Literal["scatter", "group"] = "scatter",
            required_memory_MB: int = 0,
            quiet: bool = False,
            device: Literal["cpu", "auto"] | int = "auto",
            #n_episodes: int,
            ):
        
        selected_device = get_device(device, fill_strategy, required_memory_MB)

        outputs, inputs = self.prepare_dataset(selected_device)

        assert inputs.shape[-1] == self._distilers[0].input_shape and outputs.shape[-1] == self._distilers[0].output_shape 
        #episode_len = len(distributions)//self.n_agents
        #for i in range (self.n_agents):
        #    sl = slice(i*episode_len,(i+1)*episode_len)
        if self.individual_agents:
            for ag in range(len(self._distilers)):
                dist = self._distilers[ag]
                for i in range(1):
                    dist.train_(inputs[:,ag],outputs[:,ag],i)
                dist.test_(inputs[:,ag],outputs[:,ag])
        else:
            inputs = np.reshape(np.transpose(inputs,(1,0,2)), (-1,self.n_agents,inputs.shape[2]))
            outputs = np.reshape(np.transpose(outputs,(1,0,2)), (-1,self.n_agents,outputs.shape[2]))
            dist = self._distilers[0]
            for i in range(1):
                dist.train_(inputs,outputs, i)

            dist.test_(inputs,outputs)
       
    def prepare_dataset(self, device):
        """
        Prepares the dataset used to train a distilled model, depending on the type given as argument and whether it's to be extended or not.
        """
        n_sets = 10
        targets = []
        observations = []

        # Select run, for now last run
        runs = os.listdir(self._experiment.logdir)
        runs = [run for run in runs if "run" in run]
        run = runs[-1]
        run_path = os.path.join(self._experiment.logdir,run)
        tests_path = os.path.join(run_path,"test")
        tests = os.listdir(tests_path)
        assert len(tests)>=n_sets
        tests.sort(key=int)
        for test in tests[len(tests)-(n_sets+1):-1]:
            # Load test & agent
            load_test_path = os.path.join(tests_path,test)
            self._agent.load(load_test_path)
            self._agent.to(device)
            target, obs = self.simulate_one_episode()
            targets += target
            observations += obs
        return np.array(targets), np.array(observations)
        
    # Seed can at some point be replaced to a number of epochs we want to train to note change of seed for the SDT epochs?
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
                distr, act = self._agent.get_action_distribution(obs)   # TODO: Adapt with argument
                step = self._experiment.env.step(act)
                
                # This part is very specific to my current use case: to adapt and make more generic!!
                temp = obs.data
                
                distributions.append(distr)
                f_obs, ag_pos = flatten_observation(temp, self.n_agents, axis=1)  # Very specific to flattened layers. If extras also adds agent position
                if self.extras: f_obs = np.concatenate([f_obs.reshape(self.n_agents,-1), obs.extras, ag_pos], axis=1)
                else: f_obs = f_obs.reshape(self.n_agents,-1)
                observations.append(f_obs) # axis one because axis 0 is agent
                
                is_finished = step.done | step.truncated
            return distributions, observations