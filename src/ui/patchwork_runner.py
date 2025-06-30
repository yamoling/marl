import cv2
import base64
import numpy as np
from marl.models import Experiment, ReplayEpisode
from marl.distilers.sdt import SoftDecisionTree
import typed_argparse as tap

import pathlib
import os

import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button, CheckButtons
import matplotlib.cm as cm

class FrameViewer:
    
    frames: list[str]
    frame_idx: int
    selected_agent: str

    fig: plt.Figure
    ax_img: plt.Axes

    control_fig: plt.Figure
    btn_prev: Button
    btn_next: Button

    ctrl_height: int


    def __init__(self, frames: list[str]):
        self.frames = frames
        self.frame_idx = 0
        self.selected_agent = 'Agent 0'

        self.init_plots()    
        self.init_ctrl()

    def init_plots(self):
        # Main Plot Figure
        self.fig, self.ax_img = plt.subplots(1, 1, figsize=(10, 6))
        self.fig.canvas.manager.set_window_title("Frame Viewer")

    def init_ctrl(self):
         # Control Panel (Separated, to more easily save the actual plot)
        self.control_fig = plt.figure(figsize=(3, 4))
        self.control_fig.canvas.manager.set_window_title("Controls")

        self.ctrl_height = 0.05
        self.btn_prev = Button(self.control_fig.add_axes([0.1, self.ctrl_height, 0.35, 0.1]), '←')
        self.btn_next = Button(self.control_fig.add_axes([0.55, self.ctrl_height, 0.35, 0.1]), '→')
        self.ctrl_height += 0.15

        self.btn_prev.on_clicked(self.on_prev)
        self.btn_next.on_clicked(self.on_next)

    def show(self):
        self.render()
        plt.show()

    def decode_b64_image(self, base64_str: str) -> np.ndarray:
        image = decode_b64_image(base64_str)
        image = image[:, :, ::-1]
        return image

    def update_canvas(self) -> tuple[int,int]:
        # Get data to plot
        frame = self.frames[self.frame_idx]
        img_rgb = self.decode_b64_image(frame)
        H_img, W_img = img_rgb.shape[:2]

        # Insert image and plot heatmap
        self.ax_img.imshow(img_rgb)
        self.ax_img.set_title(f"Frame {self.frame_idx + 1}/{len(self.frames)}")
        self.ax_img.axis('off')

        return H_img, W_img

    def clear_canvas(self):
        self.ax_img.clear()

    def render(self):
        """Clears the canvas, updates it and redraws it. clear and update may be overriden, this serves to abstract."""
        self.clear_canvas()
        self.update_canvas()
        self.fig.canvas.draw_idle()

    def on_next(self, event):
        """Callback of next button: Increments current index to get the frame and rerenders plot"""
        self.frame_idx = (self.frame_idx + 1) % len(self.frames)
        self.render()

    def on_prev(self, event):
        """Callback of next button: Decrements current index to get the frame and rerenders plot"""
        self.frame_idx = (self.frame_idx - 1) % len(self.frames)
        self.render()

class XFrameViewer(FrameViewer):
    
    radio_ax: plt.Axes
    radio: RadioButtons

    def __init__(self, frames: list[str]):
        super(XFrameViewer, self).__init__(frames)
        pass

    def init_ctrl(self):
        super().init_ctrl()

        self.radio_ax = self.control_fig.add_axes([0.1, self.ctrl_height, 0.8, 0.3]) # TODO: Scale with agent number (OPT)
        self.ctrl_height += 0.35
        self.radio = RadioButtons(self.radio_ax, ['Option 1', 'Option 2', 'Option 3', 'Option 4']) # TODO: Replace with agents
        self.radio.on_clicked(self.on_radio)

    def on_radio(self, label):
        """Callback of radiobuttons: Stores the new label for agent selection and rerenders plot"""
        self.selected_agent = label
        self.render()


class HeatmapXFrameViewer(XFrameViewer):

    ax_bar: plt.Axes

    check_ax: plt.Axes
    heatmap_check: CheckButtons

    show_heatmap: bool = False

    def __init__(self, frames: list[str]):
        super(HeatmapXFrameViewer, self).__init__(frames)
        pass

    def init_plots(self):
        # Main Plot Figure
        self.fig, (self.ax_img, self.ax_bar) = plt.subplots(1, 2, figsize=(10, 6))
        self.fig.canvas.manager.set_window_title("Frame Viewer")

    def init_ctrl(self):
        super().init_ctrl()

        self.check_ax = self.control_fig.add_axes([0.1, self.ctrl_height, 0.8, 0.1]) # TODO: Scale with agent number (OPT)
        self.ctrl_height += 0.1
        self.heatmap_check = CheckButtons(self.check_ax, ["Heatmap"])
        self.heatmap_check.on_clicked(self.on_check)

    def update_canvas(self):
        H_img, W_img = super().update_canvas()
        # Get data to plot
        heatmap = self.get_heatmap_data(self.selected_agent)
        bar_vals = self.get_barplot_data(self.selected_agent)

        # Overlay heatmap
        self.ax_img.imshow(cm.jet(heatmap)[:, :, :3],
                           alpha=0.33,
                           extent=(0, W_img, H_img, 0), # origin top-left, match image coords 
                           interpolation='nearest')

        # Plot barplot (extras)
        self.ax_bar.barh(np.arange(len(bar_vals)), bar_vals)
        self.ax_bar.set_title(f"Barplot: {self.selected_agent}")

    def clear_canvas(self):
        super().clear_canvas()
        self.ax_bar.clear()

    def get_heatmap_data(self, selection):
        return np.random.rand(12, 13)  # TODO: Get data from sdt/saliency

    def get_barplot_data(self, selection):
        return np.random.rand(10)  # TODO: Get data from sdt

    def on_check(self, label):
        """Callback of CheckButtons: Flips the boolean value of show_heatmap"""
        self.show_heatmap = not(self.show_heatmap)
        self.render()

LOG_PATH = pathlib.Path("logs")

def decode_b64_image(base64_str: str) -> np.ndarray:
    if base64_str is None:
        return ""
    base64_str = base64_str.split(',')[-1]
    image_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


def get_selection(cur_dir: str, file_list: str) -> pathlib.Path:
    print("Select one of the following files (you may use the list index):")
    print(file_list)
    file_str = input()
    if file_str in  file_list:
        file_path = cur_dir / file_str
    elif file_str.lstrip("-").isnumeric() and abs(int(file_str)) < len(file_list):
        file_str = file_list[int(file_str)]
        file_path = cur_dir / file_str
    else:
        raise Exception("Not a valid file")
    return file_path

def handle_selection() -> tuple[Experiment, pathlib.Path, pathlib.Path]:
    exp_list = os.listdir(LOG_PATH)

    exp_path = get_selection(LOG_PATH, exp_list)     
    experiment = Experiment.load(exp_path)
    if experiment.logdir != exp_path:
            experiment.logdir = exp_path
    print()

    exp_cont = os.listdir(exp_path)
    run_list = [run for run in exp_cont if "run" in run]
    run_path = get_selection(exp_path, run_list)
    print()

    test_path = run_path / "test"
    test_list = os.listdir(test_path)
    timestep_path = get_selection(test_path, test_list)
    print()

    distil_path = None
    if "distil" in exp_cont:
        distil_list = [distil for distil in os.listdir(exp_path/"distil") if "distil" in distil]
        if len(distil_list) > 0:
            print("Distillation(s) of the experiment's model found, do you want to visualize it? \n[Y/N]")
            yn = input()
            if str.upper(yn) == "Y": distil_path = get_selection(exp_path/"distil", distil_list)

    return experiment, timestep_path, distil_path

def handle_distillation(episode: ReplayEpisode, distil_path: pathlib.Path):
    if "sdt" in str(distil_path):
        distiller = SoftDecisionTree.load(distil_path)
    else: raise Exception(f"Distiller {distil_path} not implemented in visualization yet.")
    print()


def main():
    print("Episode runner")

    experiment, timestep_path, distil_path = handle_selection()

    episode_str = timestep_path / "0"
    episode = experiment.replay_episode(episode_str)
    episode.episode.rewards

    if distil_path is not None: 
        handle_distillation(episode, distil_path)
        viewer = HeatmapXFrameViewer(episode.frames)
    else:
        viewer = FrameViewer(episode.frames)
    viewer.show()

if __name__ == "__main__":
    main()