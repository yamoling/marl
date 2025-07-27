import numpy as np
from marl.models import Experiment

import numpy as np
import matplotlib.pyplot as plt

def plot_target_distro(targets, path, labels):
    n_agents  = targets.shape[1]
    n_classes = len(labels)                     # 0‒4
    # counts[i, j] = how many times class j occurs for agent i
    counts = np.stack([
        np.bincount(np.argmax(targets,axis=-1)[:, i], minlength=n_classes)
        for i in range(n_agents)
    ])
    fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True)

    bottom = np.zeros(n_agents)       # where the next bar segment starts

    for cls in range(n_classes):
        ax.bar(
            np.arange(n_agents),
            counts[:, cls],
            bottom=bottom,
            label=labels[cls]
        )
        bottom += counts[:, cls]

    ax.set_xlabel('Agent index')
    ax.set_ylabel('Number of targets')
    ax.set_title('Target distribution per agent')
    ax.set_xticks(np.arange(n_agents),[f"Agent {id}" for id in range(n_agents)])
    ax.legend(title='Target class', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(path)

def plot_reference_lines(dataset):
    # Calculate stats
    mean_val = np.mean(dataset)
    median_val = np.median(dataset)
    top25_thresh = np.percentile(dataset, 75)
    top10_thresh = np.percentile(dataset, 90)

    # Add lines for mean, median, 75th percentile 90th percentile
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f"Mean = {mean_val:.3f}")
    plt.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f"Median = {median_val:.3f}")
    plt.axvline(top25_thresh, color='purple', linestyle='--', linewidth=2, label=f"75th %ile = {top25_thresh:.3f}")
    plt.axvline(top10_thresh, color='yellow', linestyle='--', linewidth=2, label=f"90th %ile = {top10_thresh:.3f}")

def plot_importance(importance_scores,path):

    # Plot histogram
    plt.figure(figsize=(10, 5))
    plt.hist(importance_scores, bins=75, alpha=0.7, color='skyblue', edgecolor='black')
    # Plot: mean, median, percentiles... reference lines
    plot_reference_lines(importance_scores)

    # Labels
    plt.title("Distribution of State Importance Scores")
    plt.xlabel("Importance Score")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)

def plot_importance_with_targets(importance_scores,        # shape (n_samples,)
                                    label_probs,              # shape (n_samples, 5)
                                    path,
                                    labels,
                                    bins=50,
                                    agg="mean"):               # "sum" or "mean"
    # Histogram of `importance_scores` with stacked bar segments that
    # visualise the aggregated action probabilities per bin.
    
    importance_scores = np.asarray(importance_scores).ravel()
    label_probs       = np.asarray(label_probs)
    assert label_probs.shape[0] == importance_scores.size and label_probs.shape[1] == 5, \
        "`label_probs` must be (n_samples, 5)"

    # Bin edges & indices

    _, bin_edges = np.histogram(importance_scores, bins=bins)
    bin_width    = np.diff(bin_edges)
    bin_centers  = bin_edges[:-1] + bin_width/2
    bin_idx      = np.digitize(importance_scores, bin_edges[:-1], right=False) - 1
    bin_idx      = np.clip(bin_idx, 0, bins-1)              # safety

    # Aggregate probabilities per bin

    agg_matrix = np.zeros((bins, 5))                        # (bins × classes)
    for b in range(bins):
        mask = bin_idx == b
        if mask.any():
            if agg == "sum":
                agg_matrix[b] = label_probs[mask].sum(axis=0)
            elif agg == "mean":
                agg_matrix[b] = label_probs[mask].mean(axis=0) #* mask.sum()
            else:
                raise ValueError("agg must be 'sum' or 'mean'")

    # Stacked bar plot

    class_colors = ["#4C72B0", "#55A868", "#C44E52",
                    "#8172B3", "#CCB974"]      

    fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True)
    bottom = np.zeros(bins)
    for c in range(5):
        ax.bar(bin_centers,
               agg_matrix[:, c],
               bottom=bottom,
               width=bin_width,
               color=class_colors[c],
               edgecolor="black",
               align="center",
               label=labels[c])
        bottom += agg_matrix[:, c]

    # Plot: mean, median, percentiles... reference lines
    plot_reference_lines(importance_scores)

    # Cosmetics & save
    ax.set_title("Action distribution by importance bin")
    ax.set_xlabel("Importance score")
    ax.set_ylabel("Expected count" if agg == "sum" else "Scaled count")
    ax.grid(True, axis="y", alpha=.3)
    ax.legend(ncol=2, bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0)
    plt.savefig(path, dpi=1000, bbox_inches="tight")
    plt.close()

def get_agent_pos(observations: np.ndarray):
    n_timesteps, n_agents, _, height, width = observations.shape
    agent_positions = np.zeros((n_timesteps, n_agents, 2), dtype=int)
    # Use advanced indexing to extract each agent’s own layer
    for a in range(n_agents):
        # Get the agent’s own layer across all timesteps
        agent_layer = observations[:, a, a]
        pos = np.argwhere(agent_layer == 1)
        # Store each position
        for t, i, j in pos:
            agent_positions[t, a] = [i, j]

    return agent_positions

def get_env_infos(experiment: Experiment):
    env = experiment.env
    action_names = env.action_space.action_names
    extras_meanings = env.extras_meanings

    return action_names, extras_meanings

def flatten_observation(observation, n_agents, axis=0):
    """Flattens the observation as per the structure of a layered observation from LLE.
    axis is the axis starting which the field is. Note that in the case of layered, axis=1 is the one representing the type of observation. 
    """
    agent_pos = np.zeros((n_agents,2))
    observation = np.array(observation)
    flattened_obs = np.full((n_agents, observation.shape[axis+1], observation.shape[axis+2]), -1, dtype=int)
    # Clone to avoid modifying the original observation
    obs_a = np.copy(observation)
    laser_0 = n_agents + 1
    agent_pos[0] = np.argwhere(obs_a[0,0]==1) # Store agent position for agent 0

    for a_idx in range(1,n_agents): # Change perspective of agents 1 to 3
        # Swap agent layer of agent a_idx
        obs_a[a_idx, [0, a_idx]] = observation[a_idx, [a_idx, 0]]
        # Store agent position for subsequent agents
        agent_pos[a_idx] = np.argwhere(obs_a[a_idx,0]==1)
        # Swap laser layer of agent a_idx, where laser_i = n_agents+1+i
        laser_a_idx = laser_0 + a_idx
        obs_a[a_idx, [laser_0, laser_a_idx]] = observation[a_idx, [laser_a_idx, laser_0]]

    # Find the first n (axis 0) where O[n, i, j] == 1
    # This gives a mask of the same shape as O
    mask = obs_a == 1
    # Get the first 'n' where the condition is met along axis 
    first_n = np.argmax(mask, axis=axis, )
    # Check if *any* 1 was found along axis 0 for each (i, j)
    any_valid = mask.any(axis=axis)
    # Only update F where a 1 was found
    flattened_obs[any_valid] = first_n[any_valid]

    return flattened_obs+1, agent_pos # +1 to have 0 as empty cells and no overlap with agent 0 identifier
    
def abstract_observation(obs, n_agents):
    A, L, H, W = obs.shape
    LASER_START = A + 1
    LASER_END   = L - 2 # exclusive ( ‑2 = gems, ‑1 = exits )

    # Extract global elements layers
    wall_mask = obs[:, A, :, :] == 1 # (A,H,W)
    gem_mask = obs[:, -2, :, :] == 1 # (A,H,W)
    exit_mask = obs[:, -1, :, :] == 1 # (A,H,W)

    # build global elements mask (same for all agents)
    walls_yx = np.argwhere(wall_mask[0])
    gems_yx  = np.argwhere(gem_mask[0])
    exits_yx = np.argwhere(exit_mask[0])

    # laser sources & beams
    beam_mask_global = np.zeros((H, W), dtype=bool)
    laser_srcs_yx = []
    for k in range(LASER_START, LASER_END):
        layer = obs[0, k] # same for every agent
        laser_srcs_yx.extend(np.argwhere(layer == -1))
        beam_mask_global |= (layer == 1)

    # pre‑compute agent positions
    agent_pos = np.zeros((A, 2), dtype=int)  # (y,x)
    for a in range(A):
        y, x = np.argwhere(obs[a, a] == 1)[0]
        agent_pos[a] = (y, x)

    # Agent-wise feature extraction (lasers, other agents)
    labels = feature_labels(n_agents, len(gems_yx), len(laser_srcs_yx))
    features = []
    for i in range(A):
        y_i, x_i = agent_pos[i]
        f = []

        # current agent absolute position normalised [0,1]
        f += [x_i / (W-1), y_i / (H-1)]

        # relative position & euclidean distance to other agents
        for j in range(A):
            if j == i: continue
            y_j, x_j = agent_pos[j]
            dx, dy = x_j - x_i, y_j - y_i
            f += [dx, dy, np.hypot(dx, dy)]

        # relative position to every gem
        for (y_g, x_g) in gems_yx:
            f += [x_g - x_i, y_g - y_i]

        # relative position to every laser source
        for (y_l, x_l) in laser_srcs_yx:
            f += [x_l - x_i, y_l - y_i]

        # relative position to closest exit
        if len(exits_yx):
            dists = [np.hypot(x_e - x_i, y_e - y_i) for (y_e, x_e) in exits_yx]
            idx   = int(np.argmin(dists))
            y_e, x_e = exits_yx[idx]
            dx_e, dy_e = x_e - x_i, y_e - y_i
            f += [dx_e, dy_e, dists[idx]]
        else:
            f += [0, 0, 0]

        # wall distance in four cardinal directions
        up    = np.min([y_i - y for (y, x) in walls_yx if x == x_i and y < y_i], initial=H)
        down  = np.min([y - y_i for (y, x) in walls_yx if x == x_i and y > y_i], initial=H)
        left  = np.min([x_i - x for (y, x) in walls_yx if y == y_i and x < x_i], initial=W)
        right = np.min([x - x_i for (y, x) in walls_yx if y == y_i and x > x_i], initial=W)
        f += [up, down, left, right]

        # TODO: laser threat indicators
        

    return np.asarray(features), labels   # shape (n_agents, feature_dim)

def feature_labels(n_agents, n_gems, n_lasers):
    """ Compute labels for features extracted in abstract_observation
    """
    labels = []

    # Own absolute
    labels += ["Own normalized x", "Own normalized y"]

    # Relative and euclidian distance for each other agent
    for j in range(n_agents - 1):
        labels += [f"Delta x to agent {j+1}",
                   f"Delta y to agent {j+1}",
                   f"Distance to agent {j+1}"]

    # Relative distance to gems
    for g in range(n_gems):
        labels += [f"Delta x to gem{g+1}",
                   f"Delta y to gem{g+1}"]

    # Relative distance to laser sources
    for l in range(n_lasers):
        labels += [f"Delta x to laser {l+1}",
                   f"Delta y to laser {l+1}"]

    # Relative distance to closest exit
    labels += ["Delta x to closest exit",
               "Delta y to closest exit",
               "Distance to closest exit"]

    # wall distances (cardinal)
    labels += ["Closest wall up", "Closest wall down", "Closest wall left", "Closest wall right"]

    # TODO: laser threats

    return labels

