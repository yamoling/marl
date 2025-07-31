import cv2
import base64

import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button, CheckButtons
from matplotlib.colors import Normalize, Colormap
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.colorbar import Colorbar
from matplotlib.patches import Rectangle
import mplcursors 
from cycler import cycler

import numpy as np

from typing import Optional

def decode_b64_image(base64_str: str) -> np.ndarray:
    if base64_str is None:
        return ""
    base64_str = base64_str.split(',')[-1]
    image_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

class FrameViewer:
    frames: list[str]
    frame_idx: int
    episode_len: int
    selected_agent: str
    n_agents: int

    fig: plt.Figure
    ax_img: plt.Axes

    control_fig: plt.Figure
    btn_prev: Button
    btn_next: Button

    ctrl_height: int

    def __init__(self, frames: list[str], n_agents: int=1):
        self.frames = frames
        self.frame_idx = 0
        self.episode_len = len(frames)
        self.n_agents = n_agents

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

        # Insert image
        self.ax_img.imshow(img_rgb)
        self.ax_img.set_title(f"Frame {self.frame_idx + 1}/{len(self.frames)}")
        self.ax_img.axis('off')

        return H_img, W_img

    def clear_canvas(self):
        self.ax_img.clear()

    def render(self):
        """Clears the canvas, updates it and redraws it. clear and update may be overridden, this serves to abstract."""
        self.clear_canvas()
        self.update_canvas()
        self.fig.canvas.draw_idle()

    def on_next(self, event):
        """Callback of next button: Increments current index to get the frame and rerenders plot"""
        self.frame_idx = (self.frame_idx + 1) % (len(self.frames))
        self.render()

    def on_prev(self, event):
        """Callback of next button: Decrements current index to get the frame and rerenders plot"""
        self.frame_idx = (self.frame_idx - 1) % len(self.frames)
        self.render()

class ActFrameViewer(FrameViewer):   
    radio_ax: plt.Axes
    radio: RadioButtons
    selected_agent: str
    n_agents: int
    agent_ids: dict = {}

    agent_pos: np.ndarray
    actions:  np.ndarray[np.ndarray,np.ndarray]
    action_names = list[str]
    fig_action: plt.Figure
    ax_action: plt.Axes

    qvalues_check_ax: Optional[plt.Axes] = None
    qvalues_check: Optional[CheckButtons] = None
    qvalues: Optional[np.ndarray] = None
    qvalue_labels: Optional[list[str]] = None
    show_qvalues: bool = False
    cursor: mplcursors.cursor = None

    def __init__(self, frames: list[str], n_agents: int, agent_pos: np.ndarray, actions: np.ndarray[np.ndarray,np.ndarray], action_names: list[str], qvalues: Optional[np.ndarray] = None, qvalue_labels: Optional[list[str]] = None):
        for i in range(n_agents):
            self.agent_ids[f"Agent {i}"] = i
        self.qvalues = qvalues # Before parent init, because else qvalues = None

        super(ActFrameViewer, self).__init__(frames, n_agents)
        # Assert specific to sdt case, so comment for now til a better implementation
        #assert actions.shape[:-1] == (self.episode_len-1,n_agents,2) or actions.shape[:-1] == (self.episode_len-1,n_agents) # episode_len, based on len(frames), but there is 1 more frame at the end state, which has no related step

        self.selected_agent = next(iter(self.agent_ids))
        self.selected_agent_id = self.agent_ids[self.selected_agent]
        self.actions = actions
        self.action_names = action_names
        self.agent_pos = agent_pos
        self.qvalue_labels = qvalue_labels
        self.init_action_plot()

    def init_ctrl(self):
        super().init_ctrl()
        elem_height = 0.075*self.n_agents
        self.radio_ax = self.control_fig.add_axes([0.1, self.ctrl_height, 0.8, elem_height])
        self.ctrl_height += elem_height+0.05
        self.radio = RadioButtons(self.radio_ax, list(self.agent_ids.keys())) # TODO: Replace with agents
        self.radio.on_clicked(self.on_radio)

        if self.qvalues is not None:
            self.qvalues_check_ax = self.control_fig.add_axes([0.1, self.ctrl_height, 0.8, 0.1])
            self.ctrl_height += 0.15
            self.qvalues_check = CheckButtons(self.qvalues_check_ax, ["Qvalues"])
            self.qvalues_check.on_clicked(self.on_check_qvalues)


    def init_action_plot(self):
        """Initializes a separate figure to show the action distribution of the selected agent."""
        self.fig_action, self.ax_action = plt.subplots(figsize=(4, 4))
        self.fig_action.canvas.manager.set_window_title("Action Distribution")
    
    def init_qvalues_plot(self):
        """Initializes a separate figure to show the decomposed qvalues distribution of the selected agent."""
        self.fig_action, self.ax_action = plt.subplots(figsize=(4, 4))
        self.fig_action.canvas.manager.set_window_title("Decomposed Qvalues Distribution")

    def update_canvas(self):
        H_img, W_img = super().update_canvas()
        if self.frame_idx < self.episode_len:
            # = Plot action distribution
            self.ax_action.clear()
            x = np.arange(len(self.action_names))

            if not self.show_qvalues:
                dists = self.get_distribution()
                if len(dists) == 1:
                    self.ax_action.bar(x, dists[0], color='skyblue', label='Policy')
                else:
                    self.ax_action.bar(x - 0.2, dists[0], width=0.4, label='Distilled', color='skyblue')
                    self.ax_action.bar(x + 0.2, dists[1], width=0.4, label='Policy', color='orange')
                self.ax_action.set_ylim(0, 1)
                self.ax_action.legend()
            else:
                qvalues = self.get_qvalues()
                n_acts, n_qvals = qvalues.shape
                default_cycler = plt.rcParams["axes.prop_cycle"]
                colours = (default_cycler * cycler(linestyle=["-"])).by_key()["color"][:n_qvals]
                pos_bottom = np.zeros(n_acts)       #  Keep track of stack positions
                neg_bottom = np.zeros(n_acts)

                for k in range(n_qvals):
                    vals = qvalues[:, k]
                    # Consider positive and negative values separately
                    pos_vals = np.where(vals >= 0, vals, 0)
                    neg_vals = np.where(vals <  0, vals, 0)

                    bars_pos = self.ax_action.bar(range(n_acts), 
                            pos_vals,              
                            bottom=pos_bottom,
                            color=colours[k], 
                            label=self.qvalue_labels[k])
                    self._attach_cursor(bars_pos, [f"{self.qvalue_labels[k]}  +{v:.3f}" for v in pos_vals])
                    bars_neg = self.ax_action.bar(range(n_acts),
                            neg_vals,
                            bottom=neg_bottom,
                            color=colours[k],)
                    self._attach_cursor(bars_neg, [f"{self.qvalue_labels[k]}  {v:.3f}"  for v in neg_vals])

                    pos_bottom += pos_vals
                    neg_bottom += neg_vals
                ymin = neg_bottom.min() # often ≤ 0
                ymax = pos_bottom.max()
                self.ax_action.set_ylim(ymin * 1.1, ymax * 1.1)
                self.ax_action.set_ylabel("Q-value")
                self.ax_action.legend()

            self.ax_action.set_title(f"{"Actions" if not self.show_qvalues else "Q-values"} - {self.selected_agent}, Frame {self.frame_idx + 1}")
            self.ax_action.axhline(0, color="k", linewidth=0.8)
            self.ax_action.set_xticks(x)
            self.ax_action.set_xticklabels(self.action_names)
        else:
            self.ax_action.clear()
            self.ax_action.set_title(f"Frame {self.frame_idx + 1}")
        self.fig_action.canvas.draw_idle()
        return H_img, W_img
    
    def _attach_cursor(self, container, texts):
        """Give mplcursors <container> and show <texts[i]> for bar i on hover."""
        self.cursor = mplcursors.cursor(container, hover=mplcursors.HoverMode.Transient)
        @self.cursor.connect("add")
        def _(sel):
            idx = sel.index # bar number inside the container
            sel.annotation.set_text(texts[idx])

    def get_distribution(self):
        """Returns one or two action distributions depending on the action array shape."""
        vals = self.actions[self.frame_idx, self.selected_agent_id]
        return [vals]
        
    def get_qvalues(self):
        """Returns the array of qvalues for each action for the current agent and time-step."""
        return self.qvalues[self.frame_idx, self.selected_agent_id]
        
    def get_agent_pos(self):
        return self.agent_pos[self.frame_idx, self.selected_agent_id]

    def on_radio(self, label):
        """Callback of radiobuttons: Stores the new label for agent selection and rerenders plot"""
        self.selected_agent = label
        self.selected_agent_id = self.agent_ids[self.selected_agent]
        self.render()

    def on_check_qvalues(self, label):
        """Callback of Qvalue CheckButtons: Flips the boolean value of show_qvalues"""
        self.show_qvalues = not(self.show_qvalues)
        self.render()

class HeatmapActFrameViewer(ActFrameViewer):
    ax_bar: plt.Axes

    check_ax: plt.Axes
    heatmap_check: CheckButtons

    show_heatmap: bool = True
    heatmap_dat: np.ndarray
    extras_dat: np.ndarray
    heatmap_dat: list[str]
    extras: bool = False

    heatmap_idx: int = 0
    heatmap_layer: int = 1
    heatmap_layered: bool = True
    heatmap_next: Button = None
    heatmap_prev: Button = None

    color_bar: Colorbar
    norm_layers: list[Normalize]
    cmap: Colormap

    def __init__(self, frames: list[str], n_agents: int, agent_pos: np.ndarray, actions: np.ndarray, action_names: list[str], heatmap_dat: np.ndarray, extras_dat: np.ndarray, extras_meaning: list[str], qvalues: Optional[np.ndarray] = None, qvalue_labels: Optional[list[str]] = None):
        self.heatmap_dat = heatmap_dat
        self.extras_dat = extras_dat
        self.extras_meaning = extras_meaning + ["Agent pos x", "Agent pos y"]
        if len(heatmap_dat.shape[2:]) == 3: # Extra layer to traverse heatmaps, i.e. hierarchical filters of sdt
            self.heatmap_layer = heatmap_dat.shape[2]
            self.heatmap_layered = True
        elif len(heatmap_dat.shape[2:]) == 2: # Simple heatmap
            pass
        else: raise Exception(f"Heatmap data of dimension {self.heatmap_dat.shape} not supported!")

        super(HeatmapActFrameViewer, self).__init__(frames,n_agents,agent_pos,actions,action_names,qvalues,qvalue_labels)

        self.extras = self.extras_dat is not None
        if self.extras: assert heatmap_dat.shape[:2] == (self.episode_len-1,self.n_agents) and extras_dat.shape[:2] == (self.episode_len-1,self.n_agents) # episode_len-1, because extra frame for final state
        else: assert heatmap_dat.shape[:2] == (self.episode_len-1,self.n_agents)

    def init_plots(self):
        # Main Plot Figure
        self.fig, (self.ax_img, self.ax_bar) = plt.subplots(1,2, figsize=(10, 6), gridspec_kw={'width_ratios': [7, 3]})
        self.fig.subplots_adjust(wspace=0.3)
        self.ax_bar.yaxis.set_tick_params(pad=10)

        self.fig.canvas.manager.set_window_title("Frame Viewer")

        # Shared colorbar
        # Normalize based on combined data range
        self.norm_layers = []

        if self.heatmap_layered:
            if self.extras:
                for i in range(self.heatmap_layer):
                    hm_layer_data = self.heatmap_dat[:, :, i]
                    ex_layer_data = self.extras_dat[:, :, i]
                    vmin = min(np.min(hm_layer_data), np.min(ex_layer_data))
                    vmax = max(np.max(hm_layer_data), np.max(ex_layer_data))
                    self.norm_layers.append(Normalize(vmin=vmin, vmax=vmax))
            else:
                for i in range(self.heatmap_layer):
                    layer_data = self.heatmap_dat[:, :, i]
                    vmin = np.min(layer_data)
                    vmax = np.max(layer_data)
                    self.norm_layers.append(Normalize(vmin=vmin, vmax=vmax))
        else:
            if self.extras:
                vmin = min(np.min(self.heatmap_dat), np.min(self.extras_dat))
                vmax = max(np.max(self.heatmap_dat), np.max(self.extras_dat))
            else:
                vmin = np.min(self.heatmap_dat)
                vmax = np.max(self.heatmap_dat)
            self.norm_layers.append(Normalize(vmin=vmin, vmax=vmax))  # single norm

        self.cmap = get_cmap('coolwarm')
        sm = ScalarMappable(norm=self.norm_layers[0], cmap=self.cmap)
        sm.set_array([])  # required by colorbar API
        self.colorbar = self.fig.colorbar(sm, ax=[self.ax_img, self.ax_bar],
                                        orientation='vertical', shrink=0.85,
                                        pad=0.02)

    def init_ctrl(self):
        super().init_ctrl()

        self.check_ax = self.control_fig.add_axes([0.1, self.ctrl_height, 0.8, 0.1])
        self.ctrl_height += 0.15
        self.heatmap_check = CheckButtons(self.check_ax, ["Heatmap"], actives=["Heatmap"])
        self.heatmap_check.on_clicked(self.on_check_heatmap)

        if self.heatmap_layered: # TODO: As is always true at init, change way of doing if relevant/possible timewise
            self.heatmap_prev = Button(self.control_fig.add_axes([0.1, self.ctrl_height, 0.35, 0.1]), '←')
            self.heatmap_next = Button(self.control_fig.add_axes([0.55, self.ctrl_height, 0.35, 0.1]), '→')
            self.ctrl_height += 0.15

            self.heatmap_prev.on_clicked(self.on_heatmap_prev)
            self.heatmap_next.on_clicked(self.on_heatmap_next)

    def highlight_agent(self, heatmap_shape, H_img, W_img):
        #  Highlight current agent cell:
        grid_y, grid_x = self.get_agent_pos()
        # Image-space coordinates for rectangle
        H_map, W_map = heatmap_shape
        scale_x = W_img / W_map
        scale_y = H_img / H_map
        rect_x = grid_x * scale_x
        rect_y = grid_y * scale_y

        self.agent_patch = Rectangle(
            (rect_x, rect_y), scale_x, scale_y,  # (x, y, width, height)
            edgecolor='red',
            facecolor='none',
            linewidth=2
        )
        self.ax_img.add_patch(self.agent_patch)

    def update_canvas(self):
        H_img, W_img = super().update_canvas()
        if self.frame_idx < self.episode_len:
            # = Overlay heatmap
            heatmap = self.get_heatmap_data()
            if self.show_heatmap:
                norm = self.norm_layers[self.heatmap_idx if self.heatmap_layered else 0]
                self.ax_img.imshow(self.cmap(norm(heatmap))[:, :, :3],
                            alpha=0.8,
                            extent=(0, W_img, H_img, 0),  # match pixel grid
                            interpolation='nearest')
                # = Plot barplot (extras)
                if self.extras:
                    norm = self.norm_layers[self.heatmap_idx if self.heatmap_layered else 0]
                    bar_vals = self.get_barplot_data()
                    colors = self.cmap(norm(bar_vals))
                    self.ax_bar.barh(np.arange(len(bar_vals)), bar_vals, color=colors, height=0.6)
                    self.ax_bar.set_yticks(np.arange(len(self.extras_meaning)))
                    self.ax_bar.set_yticklabels(self.extras_meaning)
                    self.ax_bar.set_title(f"Barplot: {self.selected_agent}")

                    self.ax_bar.xaxis.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
                    self.ax_bar.axvline(0, color='black', linewidth=1)
            
            if self.agent_pos is not None:
                self.highlight_agent(heatmap.shape, H_img, W_img)

            if self.heatmap_layer > 1:
                self.ax_img.set_title(f"Frame {self.frame_idx + 1}/{len(self.frames)}, Filter Layer: {self.heatmap_idx+1}/{self.heatmap_layer}")

        else:
            self.ax_img.set_title(f"Frame {len(self.frames)}/{len(self.frames)}")

    def clear_canvas(self):
        super().clear_canvas()
        self.ax_bar.clear()

    def get_distribution(self): # Overload
        """Returns one or two action distributions depending on the action array shape."""
        vals = self.actions[self.frame_idx, self.selected_agent_id]
        if vals.ndim == 1:
            return [vals]
        elif vals.ndim == 2:
            return [vals[0], vals[1]]

    def get_heatmap_data(self):
        """Returns the arrays to plot the heatmap in function of the current frame idx, current selected agent and if applicable current layer of the heatmap."""
        if self.heatmap_layered: return self.heatmap_dat[self.frame_idx,self.selected_agent_id,self.heatmap_idx]
        else: return self.heatmap_dat[self.frame_idx,self.selected_agent_id]

    def get_barplot_data(self):
        """Returns the arrays to plot the extras barplot in function of the current frame idx, current selected agent and if applicable current layer of the heatmap."""
        if self.heatmap_layered: return self.extras_dat[self.frame_idx,self.selected_agent_id,self.heatmap_idx]
        else: return self.extras_dat[self.frame_idx,self.selected_agent_id]

    def on_check_heatmap(self, label):
        """Callback of CheckButtons: Flips the boolean value of show_heatmap"""
        self.show_heatmap = not(self.show_heatmap)
        self.render()

    def on_heatmap_next(self, event):
        """Callback of the heatmap next button: Increments current heatmap index to get the heatmap and rerenders plot"""
        self.heatmap_idx = (self.heatmap_idx + 1) % self.heatmap_layer
        self.render()

    def on_heatmap_prev(self, event):
        """Callback of the heatmap next button: Decrements current heatmap index to get the heatmap and rerenders plot"""
        self.heatmap_idx = (self.heatmap_idx - 1) % self.heatmap_layer
        self.render()
