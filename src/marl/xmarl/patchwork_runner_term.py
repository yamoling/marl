from marl.models import Experiment
from marlenv.models import Episode

import pathlib
import os
import numpy as np

from marl.xmarl.distilers.sdt import SoftDecisionTree
from marl.xmarl import FrameViewer, ActFrameViewer, HeatmapActFrameViewer, AbstractActFrameViewer
from marl.xmarl.distilers.utils import get_env_infos, feature_labels
from marl.xmarl import FilePickerScreen

from textual.app import App, ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import (
    Button,
    Label,
    Checkbox,
    Static,
    Switch,
)
from textual import on
from textual.reactive import reactive

from pathlib import Path


LOG_PATH = Path("logs")

class Selector(App):
    CSS = """
    Screen {
        align: center middle;
        padding: 2;
    }
    #form {
        width: 70%;
        border: round $accent;
        padding: 2;
    }
    Input, Select {
        width: 100%;
    }
    """

    spec_sep = " & "
    experiment = None
    experiment_path = Path()
    distil_path = Path()
    run_path = Path()
    test_path = Path()
    timestamp_path = Path()
    dist_type = 0

    show_extra_input = reactive(False)

    BINDINGS = [
        ("up",    "focus_previous", "Focus up"),
        ("down",  "focus_next",     "Focus down"),
        ("enter", "press_button",   "Press"),
    ]
    def action_focus_previous(self):
        self.screen.focus_previous()
    def action_focus_next(self):
        self.screen.focus_next()
    def action_press_button(self):
        # `self.focused` is the widget with focus
        if hasattr(self.focused, "pressed"):
            self.focused.press()

    def compose(self) -> ComposeResult:
        yield Static("Patchwork Runner", id="title", classes="bold")
        with Vertical(id="form"):
            with Horizontal():
                yield Button("Select Experiment", id="select_experiment")
                yield Label("No experiment selected", id="experiment_label")

            with Horizontal():
                yield Button("Select Run", id="select_run", disabled=True)
                yield Label("No run selected", id="run_label")
            
            with Horizontal():
                yield Button("Select Test", id="select_test", disabled=True)   
                yield Label("No test selected", id="test_label")

            with Horizontal(id="qvalues"):
                yield Checkbox(id="qvals_check")
                yield Label("Show decomposed Qvalues?", id="qvals_label")

            with Horizontal(id="distil_h"):
                yield Button("Select Distillation", id="select_distil", disabled=True)
                yield Label("No Distillation selected", id="distil_label")

            with Horizontal(id="distil_way"):
                yield Label("Forward", id="dist_switch_fw")
                yield Switch(id="dist_switch")
                yield Label("Backward", id="dist_switch_bw")
                

            with Horizontal():
                yield Button("Visualize", id="viz", variant="success", disabled=True)
                yield Button("Cancel", id="cancel", variant="error")


    def on_mount(self):
        self.query_one("#qvalues", Horizontal).display = False
        self.query_one("#distil_h", Horizontal).display = False
        self.query_one("#distil_way", Horizontal).display = False
    
    # == EXPERIMENT ==

    def reset_experiment(self):
        self.experiment_path = ""
        self.experiment = None
        # Handle cancellation or no file selected
        self.query_one("#experiment_label", Label).update("No experiment selected")
        self.query_one("#viz", Button).disabled = True
        self.query_one("#select_run", Button).disabled = True
        self.reset_run()

        self.query_one("#select_distil", Button).disabled = True
        self.query_one("#distil_h", Horizontal).display = False
        self.query_one("#qvalues", Horizontal).display = False
        self.reset_distil()


    def set_experiment(self, experiment):
        if experiment:
            # Reset extras: distil, qvalues, etc...
            self.reset_distil()
            self.query_one("#qvalues", Horizontal).display = False
            
            self.experiment_path = LOG_PATH / experiment  # Get the file path when popped
            self.experiment = Experiment.load(self.experiment_path)
            if self.experiment.logdir != self.experiment_path:
                self.experiment.logdir = self.experiment_path
            self.query_one("#experiment_label", Label).update(experiment)
            self.query_one("#viz", Button).disabled = False
            self.query_one("#select_run", Button).disabled = False

            if "distil" in os.listdir(self.experiment_path):
                self.query_one("#select_distil", Button).disabled = False
                self.query_one("#distil_h", Horizontal).display = True

            if self.experiment.log_qvalues:
                self.query_one("#qvalues", Horizontal).display = True
            self.reset_run()
        else:
            self.reset_experiment()

    @on(Button.Pressed, "#select_experiment")
    async def open_experiment_picker(self):
        exp_list = os.listdir(LOG_PATH)
        await self.push_screen(FilePickerScreen("Experiment", exp_list), callback=self.set_experiment)  # Push FilePickerScreen

    # == RUN ==

    def reset_run(self):
        self.run_path = ""
        # Handle cancellation or no file selected
        self.query_one("#run_label", Label).update("No Run selected")
        self.query_one("#select_test", Button).disabled = True
        self.reset_test()
        
    def set_run(self, run):
        if run:
            self.run_path = self.experiment_path / run  # Get the file path when popped
            self.query_one("#run_label", Label).update(run)
            self.query_one("#select_test", Button).disabled = False
            self.reset_test()
        else:
            self.reset_run()

    @on(Button.Pressed, "#select_run")
    async def open_run_picker(self):
        exp_cont = os.listdir(self.experiment_path)
        run_list = [run for run in exp_cont if "run" in run]
        await self.push_screen(FilePickerScreen("Run", run_list), callback=self.set_run)  # Push FilePickerScreen

    # == TEST ==

    def reset_test(self):
        self.test_path = ""
        # Handle cancellation or no file selected
        self.query_one("#test_label", Label).update("No Test selected")
        
    def set_test(self, test):
        if test:
            self.timestamp_path = self.test_path / test  # Get the file path when popped
            self.query_one("#test_label", Label).update(test)
            #self.query_one("#select_distil", Button).disabled = False
        else:
            self.reset_test()

    @on(Button.Pressed, "#select_test")
    async def open_test_picker(self):
        self.test_path = self.run_path / "test"
        test_list = sorted(os.listdir(self.test_path),key=int)
        await self.push_screen(FilePickerScreen("Run", test_list), callback=self.set_test)  # Push FilePickerScreen

    # == DISTIL ==

    def reset_distil(self):
        self.distil_path = ""
        self.extra = False
        self.abstract = False
        # Handle cancellation or no file selected
        self.query_one("#distil_label", Label).update("No Distillation selected")
        self.query_one("#select_distil", Button).disabled = True
        
    def set_distil(self, distil):
        if distil:
            if self.spec_sep in distil:
                prefix, distil = distil.split(self.spec_sep)
                self.distil_path = self.experiment_path / "distil" / prefix
            else: self.distil_path = self.experiment_path / "distil" # Get the file path when popped
            self.distiler_path = self.distil_path/distil
            self.extra = "extra" in distil
            self.abstract = "abstract" in distil
            self.query_one("#distil_label", Label).update(str(self.distil_path/distil))
            if "sdt" in distil:
                self.query_one("#distil_way", Horizontal).display = True
        else:
            self.reset_distil()

    @on(Button.Pressed, "#select_distil")
    async def open_distil_picker(self):
        distil_list = []
        for distil in os.listdir(self.experiment_path/"distil"):
            if ".pkl" in distil: distil_list.append(distil)
            elif "individual" in distil:
                for ind_distil in os.listdir(self.experiment_path/"distil"/distil):
                    if ".pkl" in ind_distil and "sdt" in ind_distil:
                        ind_dist_name = distil + self.spec_sep + ind_distil.split('_',1)[1]
                        if ind_dist_name not in distil_list: distil_list.append(ind_dist_name) # Add with special separator to split in callback
        await self.push_screen(FilePickerScreen("Distillation", distil_list), callback=self.set_distil)  # Push FilePickerScreen


    # == SUBMIT/CANCEL ==

    @on(Button.Pressed, "#viz")
    def handle_submit(self):
        self.handle_selection()

    @on(Button.Pressed, "#cancel")
    def cancel_form(self):
        self.exit()

    # == SELECTION HANDLING ==
    def handle_selection(self):
        # Load the replay episode
        episode_str = self.timestamp_path / "0"
        replay = self.experiment.replay_episode(episode_str)
        episode = replay.episode

        action_names, extras_meaning, world_shape = get_env_infos(self.experiment)

        if self.distil_path:
            if not self.abstract: distilled_filters, distilled_actions, distilled_extras, agent_pos,_,_,_ = self.handle_distillation(episode)
            else: 
                distilled_filters, distilled_actions, distilled_extras, agent_pos, obs, extras, abs_labels = self.handle_distillation(episode)
                if not self.extra: extras = None
            # Insert 7x7 obs into full board if needed
            if self.experiment.env.observation_shape[1:] == (7,7) and not self.abstract:
                agent_pos = np.array(episode.states,   dtype=int)[:,:2*episode.n_agents].reshape((episode.episode_len,episode.n_agents,2))
                n_obs = np.zeros(distilled_filters.shape[0:-2] + (12,13))
                for t in range(episode.episode_len):
                    for a in range(episode.n_agents):
                        filt = distilled_filters[t, a]
                        x, y = agent_pos[t, a]
                        x_start, y_start = x-3, y-3
                        x_end, y_end = x_start+7, y_start+7
                        x_s, x_e = max(x_start,0), min(x_end,12)
                        y_s, y_e = max(y_start,0), min(y_end,13)
                        fx_s, fx_e = x_s - x_start, x_e - x_start
                        fy_s, fy_e = y_s - y_start, y_e - y_start
                        n_obs[t,a,:,x_s:x_e,y_s:y_e] = filt[:,fx_s:fx_e,fy_s:fy_e]
                distilled_filters = n_obs
            if self.query_one("#qvals_check", Checkbox).value:
                if not self.abstract: viewer = HeatmapActFrameViewer(
                        replay.frames, world_shape, episode.n_agents, agent_pos,
                        distilled_actions, action_names,
                        distilled_filters, distilled_extras, extras_meaning,
                        np.array(replay.qvalues), self.experiment.qvalue_infos[0]
                    )
                else: viewer = AbstractActFrameViewer(
                        replay.frames, world_shape, episode.n_agents, agent_pos,
                        distilled_actions, action_names,
                        distilled_filters, obs, extras,
                        extras_meaning, abs_labels,
                        np.array(replay.qvalues), self.experiment.qvalue_infos[0]
                    )   
            else:
                if not self.abstract: viewer = HeatmapActFrameViewer(
                        replay.frames, world_shape, episode.n_agents, agent_pos,
                        distilled_actions, action_names,
                        distilled_filters, distilled_extras, extras_meaning
                    )
                else: viewer = AbstractActFrameViewer(
                        replay.frames, world_shape, episode.n_agents, agent_pos,
                        distilled_actions, action_names,
                        distilled_filters, obs, extras,
                        extras_meaning, abs_labels
                ) 
        else:
            if self.query_one("#qvals_check", Checkbox).value:
                viewer = ActFrameViewer(
                    replay.frames, world_shape, episode.n_agents, None,
                    np.array(episode.actions), action_names,
                    np.array(replay.qvalues), self.experiment.qvalue_infos[0]
                )
            else: viewer = FrameViewer(replay.frames, world_shape)
        viewer.show()

    def handle_distillation(self, episode: Episode):
        if "sdt" in str(self.distiler_path):
            dist_type = self.query_one("#dist_switch", Switch).value
            if "individual" in str(self.distiler_path):
                distilled_filters, distilled_extras, distilled_actions, agent_pos = [], [], [], []
                if self.abstract: obs, extras, labels = [], [], []
                for ag in range(episode.n_agents):
                    fname = f"ag{ag}_sdt_distil{'_extra' if self.extra else ''}{'_abstract' if self.abstract else ''}.pkl"
                    distiller = SoftDecisionTree.load(self.distil_path / fname)
                    df, de, da, ap, o, e, fx = distiller.distil_episode(episode, dist_type)
                    distilled_filters.append(df)
                    distilled_extras.append(de)
                    distilled_actions.append(da)
                    agent_pos.append(ap)
                    if self.abstract:
                        obs.append(o)
                        extras.append(e)
                        labels.append(feature_labels(episode.n_agents,len(fx[2]),fx[-1][ag].keys()))
                if not self.abstract:   # Can't because abstract inhomogenous. Should modify to use order (agent,T) in viewer if wanna make cleaner
                    # Transpose lists into arrays of shape [T, agents, ...]
                    distilled_filters = np.array(distilled_filters).swapaxes(0,1)
                    if np.any(distilled_extras):
                        distilled_extras = np.array(distilled_extras).squeeze().swapaxes(0,1)
                    else: distilled_extras = None
                    distilled_actions = np.array(distilled_actions).swapaxes(0,1)
                    agent_pos = np.array(agent_pos).swapaxes(0,1)
            else:
                distiller = SoftDecisionTree.load(self.distiler_path)
                distilled_filters, distilled_extras, distilled_actions, agent_pos,_,_,_ = distiller.distil_episode(episode, dist_type)
        else:
            raise Exception(f"Distiller {self.distiler_path} not implemented in visualization yet.")
        if not self.abstract: return distilled_filters, distilled_actions, distilled_extras, agent_pos, None, None, None # Inelegant patch to be symmetric with ind (might send data if abstract)
        else: return distilled_filters, distilled_actions, distilled_extras, agent_pos, obs, extras, labels


def main():
    import sys
    try:
        Selector().run()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
