import cv2
import base64
import numpy as np
from marl.models import Experiment, ReplayEpisode
import typed_argparse as tap

import pathlib
import os

import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, CheckButtons

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button
import matplotlib.cm as cm

class FrameViewer:
    def __init__(self, frames):
        self.frames = frames
        self.frame_idx = 0
        self.selected_agent = 'Agent 0'

        # Main Plot Figure
        self.fig, (self.ax_img, self.ax_bar) = plt.subplots(1, 2, figsize=(10, 6))
        self.fig.canvas.manager.set_window_title("Frame Viewer")
        self.render()

        # Control Panel (Separated, to more easily save the actual plot)
        self.control_fig = plt.figure(figsize=(3, 4))
        self.control_fig.canvas.manager.set_window_title("Controls")

        self.radio_ax = self.control_fig.add_axes([0.1, 0.6, 0.8, 0.3]) # TODO: Scale with agent number (OPT)
        self.radio = RadioButtons(self.radio_ax, ['Option 1', 'Option 2', 'Option 3', 'Option 4']) # TODO: Replace with agents
        self.radio.on_clicked(self.on_radio)

        self.btn_prev = Button(self.control_fig.add_axes([0.1, 0.45, 0.35, 0.1]), '←')
        self.btn_next = Button(self.control_fig.add_axes([0.55, 0.45, 0.35, 0.1]), '→')

        self.btn_prev.on_clicked(self.on_prev)
        self.btn_next.on_clicked(self.on_next)

        plt.show()

    def decode_b64_image(self, base64_str: str) -> np.ndarray:
        image = decode_b64_image(base64_str)
        image = image[:, :, ::-1]
        return image

    def get_heatmap_data(self, frame, selection):
        return np.random.rand(12, 13)  # TODO: Get data from sdt/saliency

    def get_barplot_data(self, frame, selection):
        return np.random.rand(10)  # TODO: Get data from sdt

    def render(self):
        self.ax_img.clear()
        self.ax_bar.clear()

        # Get data to plot
        frame = self.frames[self.frame_idx]
        img_rgb = self.decode_b64_image(frame)
        H_img, W_img = img_rgb.shape[:2]
        heatmap = self.get_heatmap_data(frame, self.selected_agent)
        bar_vals = self.get_barplot_data(frame, self.selected_agent)

        # Insert image and plot heatmap
        self.ax_img.imshow(img_rgb)
        self.ax_img.imshow(cm.jet(heatmap)[:, :, :3],
                           alpha=0.33,
                           extent=(0, W_img, H_img, 0), # origin top-left, match image coords 
                           interpolation='nearest')
        self.ax_img.set_title(f"Frame {self.frame_idx + 1}/{len(self.frames)}")
        self.ax_img.axis('off')

        # Plot barplot (extras)
        self.ax_bar.barh(np.arange(len(bar_vals)), bar_vals)
        self.ax_bar.set_title(f"Barplot: {self.selected_agent}")

        self.fig.canvas.draw_idle()

    def on_radio(self, label):
        """Callback of radiobuttons: Stores the new label for agent selection and rerenders plot"""
        self.selected_agent = label
        self.render()

    def on_next(self, event):
        """Callback of next button: Increments current index to get the frame and rerenders plot"""
        self.frame_idx = (self.frame_idx + 1) % len(self.frames)
        self.render()

    def on_prev(self, event):
        """Callback of next button: Decrements current index to get the frame and rerenders plot"""
        self.frame_idx = (self.frame_idx - 1) % len(self.frames)
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


def get_selection(cur_dir, file_list):
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

def handle_selection():
    exp_list = os.listdir(LOG_PATH)

    exp_path = get_selection(LOG_PATH, exp_list)     
    experiment = Experiment.load(exp_path)
    if experiment.logdir != exp_path:
            experiment.logdir = exp_path
    print()

    run_list = os.listdir(exp_path)
    run_list = [run for run in run_list if "run" in run]
    run_path = get_selection(exp_path, run_list)
    print()

    test_path = run_path / "test"
    test_list = os.listdir(test_path)
    timestep_path = get_selection(test_path, test_list)
    print()
    return experiment, timestep_path


def main():
    print("Episode runner")

    experiment, timestep_path = handle_selection()

    episode_str = timestep_path / "0"
    episode = experiment.replay_episode(episode_str)
    episode.episode.rewards

    viewer = FrameViewer(episode.frames)


def t():
    pass

if __name__ == "__main__":
    main()