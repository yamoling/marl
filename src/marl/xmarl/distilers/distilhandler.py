import os
import pathlib

from typing import Literal, Optional

import numpy as np
import torch

from marl.xmarl.distilers.sdt import SoftDecisionTree
from marl.xmarl.distilers.utils import flatten_observation, plot_importance, plot_target_distro, plot_importance_with_targets, abstract_observation
from marl.models import Experiment
from marl.agents.qlearning import DQN
from marl.utils.gpu import get_device

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class DistilHandler:
    """
    Class handling a distillation process for various distiller types and varying in-/outputs.
    Based on parameters prepares the dataset, loads the original model and creates the distilled model. Trains, tests and saves the final distilled model.
    """
    _distilers: list[SoftDecisionTree] # or sklearn DT/Randomforest?
    _experiment: Experiment
    _agent: DQN

    dist_type: str
    n_agents: int
    target_labels = list[str]
    extras: bool
    individual_agents: bool

    def __init__(self,
                 experiment: Experiment,
                 n_runs: int,
                 n_sets: int,
                 epochs: int,
                 distilers: list[SoftDecisionTree], # or sklearn DT/Randomforest?
                 dist_type: str,
                 extras: bool,
                 individual_agents: bool = False, 
    ):
        self._experiment = experiment
        self._distilers = distilers
        self._agent = experiment.agent
        if self._agent.last_qvalues is None:
            self._agent.last_qvalues = np.ndarray(0)

        self.dist_type = dist_type
        self.n_runs = n_runs
        self.n_sets = n_sets
        self.epochs = epochs
        self.n_agents = self._experiment.env.n_agents
        self.target_labels = self._experiment.env.action_space.action_names
        self.individual_agents = individual_agents
        self.extras = extras


    @staticmethod
    def create(logdir: str, n_runs: int, n_sets: int, epochs: int, distiler: str, input_type: str, extras: bool, individual_agents: bool):
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
                        extras=extras,
                        max_depth=4,
                        bs=32,
                        n_agent=experiment.env.n_agents,
                        agent_id=i
                    )
                    distilers.append(distil_model)
            else:
                distil_model = SoftDecisionTree(
                    input_shape = input_shape,
                    output_shape = experiment.env.n_actions,
                    logdir = experiment.logdir,
                    extras=extras,
                    max_depth=4,
                    bs=32,
                    n_agent=experiment.env.n_agents,
                )
                distilers.append(distil_model)
            distil_handler = DistilHandler(experiment, n_runs, n_sets, epochs, distilers, distiler, extras, individual_agents)
        else:
            raise NotImplementedError(f"Distilation not implemented for agent {experiment.agent.name}")
            # Modify to check if qvalues of experiment is true or not, if true we can access them if not no, also only do it if getting qvalues, else distribution/action should always be accessible, albeit by bypassing something
        os.makedirs(distil_model.logdir,exist_ok=True)
        return distil_handler
    
    def filter_by_importance(self, inputs, outputs, importance, percentile = 50):
        """ Filters in- and output arrays by using the importance as a metric, by only keeping the most important ones withing the percentile.
        """
        if self.individual_agents:
            filtered_inputs = []
            filtered_outputs = []
            for agent_id in range(self.n_agents):
                #agent_mask = importance[:, agent_id] >= np.median(importance[:, agent_id])
                agent_mask = importance[:, agent_id] >= np.percentile(importance[:, agent_id],percentile)
                # Select samples for this agent only
                agent_inputs = inputs[agent_mask, agent_id, :]        # shape (N_i, input_shape)
                agent_outputs = outputs[agent_mask, agent_id, :]      # shape (N_i, output_shape)
                
                filtered_inputs.append(agent_inputs)
                filtered_outputs.append(agent_outputs)
            inputs = np.array(filtered_inputs).transpose(1,0,2)
            outputs = np.array(filtered_outputs).transpose(1,0,2)
            # Plot action distributions after filter
            plot_target_distro(outputs.reshape((-1,self.n_agents,len(self.target_labels))),pathlib.Path(self._distilers[0].logdir,"dataset_filtered_target_distribution"), self.target_labels)
        else: 
            #median_mask = importance >= np.median(importance)
            median_mask = importance >= np.percentile(importance,percentile)
            inputs = inputs[median_mask] # also flattens batch and agent dims
            outputs = outputs[median_mask]
            # Plot action distributions after filter
            plot_target_distro(outputs.reshape((-1,1,len(self.target_labels))),pathlib.Path(self._distilers[0].logdir,"dataset_filtered_target_distribution"), self.target_labels)
        return inputs, outputs

    def run(self, 
            #seed: int =0,
            fill_strategy: Literal["scatter", "group"] = "scatter",
            required_memory_MB: int = 0,
            quiet: bool = False,
            device: Literal["cpu", "auto"] | int = "auto",
            #n_episodes: int,
            ):
        batch_size = self._distilers[0].batch_size

        selected_device = get_device(device, fill_strategy, required_memory_MB)
        outputs, inputs, importance = self.prepare_dataset(selected_device)

        assert inputs.shape[-1] == self._distilers[0].input_shape and outputs.shape[-1] == self._distilers[0].output_shape 

        outputs, inputs = self.filter_by_importance(inputs, outputs, importance, 90)
        
        # Determine and set batch size
        n_batches = len(inputs)//batch_size
        if self.individual_agents:
            inputs = inputs[:batch_size*n_batches].reshape(n_batches,batch_size,self.n_agents,self._distilers[0].input_shape) # inputs[:batch_size*n_batches] to fit to batch sizes
            outputs = outputs[:batch_size*n_batches].reshape(n_batches,batch_size,self.n_agents,self._distilers[0].output_shape)
        else:   # Squeeze in agents dim
            inputs = inputs[:batch_size*n_batches].reshape(n_batches,batch_size,self._distilers[0].input_shape)
            outputs = outputs[:batch_size*n_batches].reshape(n_batches,batch_size,self._distilers[0].output_shape)

        inputs_train, inputs_validation, outputs_train, outputs_validation = train_test_split(
            inputs, outputs, test_size=0.5, random_state=44, shuffle=True
        )
        inputs_validation, inputs_test, outputs_validation, outputs_test = train_test_split(
            inputs_validation, outputs_validation, test_size=0.5, random_state=44, shuffle=True
        )
        if self.individual_agents:
            for ag in range(len(self._distilers)):
                train_logs = []
                valid_logs = []
                dist = self._distilers[ag]
                best_dist = 0
                for i in range(self.epochs):
                    train_logs.append(dist.train_(inputs_train[:,:,ag],outputs_train[:,:,ag],i))
                    v_acc, v_preds = dist.test_(inputs_validation[:,:,ag],outputs_validation[:,:,ag],i)
                    valid_logs.append(v_acc)

                train_logs = np.array(train_logs)
                np.savez(pathlib.Path(f"{self._distilers[0].logdir}",f"ag{ag}_{self.dist_type}_train_logs{"_extra" if self.extras else ""}.npz"),train_logs)
                np.savez(pathlib.Path(f"{self._distilers[0].logdir}",f"ag{ag}_{self.dist_type}_valid_logs{"_extra" if self.extras else ""}.npz"),valid_logs)

                dist.load_best()
                test_logs, test_preds = dist.test_(inputs_test[:,:,ag],outputs_test[:,:,ag],best_dist)
                np.savez(pathlib.Path(f"{self._distilers[0].logdir}",f"ag{ag}_{self.dist_type}_test_logs{"_extra" if self.extras else ""}.npz"),test_logs)
                np.savez(pathlib.Path(f"{self._distilers[0].logdir}",f"ag{ag}_{self.dist_type}_test_preds{"_extra" if self.extras else ""}.npz"),test_preds)

                cm_disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(np.argmax(outputs_test[:,:,ag], axis=2).flatten(), test_preds,
                                                                                   labels=np.arange(len(self.target_labels))),
                                                                                    display_labels=self.target_labels) # outputs_test one-hot 
                cm_disp.plot()
                plt.savefig(pathlib.Path(f"{self._distilers[0].logdir}",f"ag{ag}_{self.dist_type}{"_extra" if self.extras else ""}_cm.png"))

        else:
            train_logs = []
            valid_logs = []
            dist = self._distilers[0]
            best_dist = 0
            for i in range(self.epochs):
                train_logs.append(dist.train_(inputs_train,outputs_train,i))
                v_acc, v_preds = dist.test_(inputs_validation,outputs_validation,i)
                valid_logs.append(v_acc)

            train_logs = np.array(train_logs)
            np.savez(pathlib.Path(f"{self._distilers[0].logdir}",f"{self.dist_type}_train_logs{"_extra" if self.extras else ""}.npz"),train_logs)
            np.savez(pathlib.Path(f"{self._distilers[0].logdir}",f"{self.dist_type}_valid_logs{"_extra" if self.extras else ""}.npz"),valid_logs)

            dist.load_best()
            test_logs, test_preds = dist.test_(inputs_test,outputs_test,best_dist)
            np.savez(pathlib.Path(f"{self._distilers[0].logdir}",f"{self.dist_type}_test_logs{"_extra" if self.extras else ""}.npz"),test_logs)
            np.savez(pathlib.Path(f"{self._distilers[0].logdir}",f"{self.dist_type}_test_preds{"_extra" if self.extras else ""}.npz"),test_preds)
            
            cm_disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(np.argmax(outputs_test, axis=2).flatten(), test_preds, 
                                                                               labels=np.arange(len(self.target_labels))),
                                                                               display_labels=self.target_labels) # outputs_test one-hot 
            cm_disp.plot()
            plt.savefig(pathlib.Path(f"{self._distilers[0].logdir}",f"{self.dist_type}{"_extra" if self.extras else ""}_cm.png")) 


       
    def prepare_dataset(self, device):
        """Prepares the dataset used to train a distilled model, depending on the type given as argument and whether it's to be extended or not."""
        targets = []
        observations = []
        importances = []

        # Select run, for now last run
        runs = os.listdir(self._experiment.logdir)
        runs = [run for run in runs if "run" in run]
        if self.n_runs > len(runs): self.n_runs = len(runs)
        for i in range(self.n_runs):
            run = runs[i]
            run_path = os.path.join(self._experiment.logdir,run)
            tests_path = os.path.join(run_path,"test")
            tests = [file for file in os.listdir(tests_path) if not file.startswith(".")]
            assert len(tests)>=self.n_sets
            tests.sort(key=int)
            for test in tests[len(tests)-(self.n_sets+1):-1]:
                # Load test & agent
                load_test_path = os.path.join(tests_path,test)
                self._agent.load(load_test_path)
                self._agent.to(device)
                target, obs, imp = self.simulate_one_episode()
                targets += target
                observations += obs
                importances += imp
        importances = np.array(importances)
        plot_importance(importances.flatten(),pathlib.Path(self._distilers[0].logdir,"dataset_importance"))
        plot_target_distro(np.array(targets),pathlib.Path(self._distilers[0].logdir,"dataset_target_distribution"), self.target_labels)
        plot_importance_with_targets(importances.flatten(),np.array(targets).reshape(-1,len(self.target_labels)),pathlib.Path(self._distilers[0].logdir,"dataset_target_importance"), self.target_labels)
        return np.array(targets), np.array(observations), importances
        
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
            importances = []
            i = 0
            is_finished = False
            while not is_finished:
                if i == 0: obs, state = self._experiment.env.reset()
                else: obs = step.obs
                i += 1
                
                qvalues = self._agent.qnetwork.qvalues(obs).numpy(force=True)
                if self._agent.qnetwork.is_multi_objective:
                    qvalues = qvalues.sum(axis=-1)
                importances.append(qvalues.max(axis=-1) - qvalues.min(axis=-1)) # Based on idea used in MAVIPER, but since out "value" function is not specified or either the mean or sum of best qvalues, this is the same

                action = self._agent.test_policy.get_action(qvalues, obs.available_actions)

                qv_distr = torch.softmax(torch.from_numpy(qvalues), axis=1)
                

                step = self._experiment.env.step(action)
                
                
                # This part is very specific to my current use case: to adapt and make more generic!!
                temp = obs.data
                
                distributions.append(qv_distr)
                #f_obs, ag_pos = flatten_observation(temp, self.n_agents, axis=1)  # Very specific to flattened layers. If extras also adds agent position
                f_obs = abstract_observation(temp, self.n_agents)  # Very specific to flattened layers. If extras also adds agent position
                if self.extras: f_obs = np.concatenate([f_obs.reshape(self.n_agents,-1), obs.extras, ag_pos], axis=1)
                else: f_obs = f_obs.reshape(self.n_agents,-1)
                observations.append(f_obs) # axis one because axis 0 is agent
                
                is_finished = step.done | step.truncated
            return distributions, observations, importances