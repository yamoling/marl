import numpy as np
from marl.models import Experiment

import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

import math

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
    top5_thresh = np.percentile(dataset, 95)
    top3_thresh = np.percentile(dataset, 97)

    # Add lines for mean, median, 75th percentile 90th percentile
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f"Mean = {mean_val:.3f}")
    plt.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f"Median = {median_val:.3f}")
    plt.axvline(top25_thresh, color='purple', linestyle='--', linewidth=2, label=f"75th %ile = {top25_thresh:.3f}")
    plt.axvline(top5_thresh, color='yellow', linestyle='--', linewidth=2, label=f"95th %ile = {top5_thresh:.3f}")
    plt.axvline(top3_thresh, color='black', linestyle='--', linewidth=2, label=f"97th %ile = {top3_thresh:.3f}")

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

def plot_importance_with_targets(importance_scores, # shape (n_samples,)
                                    label_probs,    # shape (n_samples, 5)
                                    path,
                                    labels,
                                    bins=50,
                                    agg="mean"):               # "sum" or "mean"
    # Histogram of `importance_scores` with stacked bar segments that
    # visualise the aggregated action probabilities per bin.
    
    importance_scores = np.asarray(importance_scores).ravel()
    label_probs       = np.asarray(label_probs)
    assert label_probs.shape[0] == importance_scores.size, "`label_probs` must be (n_samples, 5)"
    n_qvals = label_probs.shape[1]

    # Bin edges & indices
    _, bin_edges = np.histogram(importance_scores, bins=bins)
    bin_width    = np.diff(bin_edges)
    bin_centers  = bin_edges[:-1] + bin_width/2
    bin_idx      = np.digitize(importance_scores, bin_edges[:-1], right=False) - 1
    bin_idx      = np.clip(bin_idx, 0, bins-1)              # safety

    # Aggregate probabilities per bin
    agg_matrix = np.zeros((bins, n_qvals))    # (bins × classes)
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

    default_cycler = plt.rcParams["axes.prop_cycle"]
    class_colors = (default_cycler * cycler(linestyle=["-"])).by_key()["color"][:n_qvals]    

    fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True)
    bottom = np.zeros(bins)
    for c in range(n_qvals):
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
    """Reutrns action and extras names/meanings as well as world shape: ONLY WORKS FOR LLE level 6/7 as is (would need a proper way to get the shape)"""
    env = experiment.env
    action_names = env.action_space.action_names
    extras_meanings = env.extras_meanings
    world_shape = (12,13)

    return action_names, extras_meanings, world_shape

def flatten_observation(observation, n_agents, axis=0):
    """Flattens the observation as per the structure of a layered observation from LLE. Keeps the grid structure though.
    axis is the axis starting which the board is. Note that in the case of layered, axis=1 is the one representing the type of observation. 
    """
    observation = np.array(observation)
    flattened_obs = np.full((n_agents, observation.shape[axis+1], observation.shape[axis+2]), -1, dtype=int)
    # Clone to avoid modifying the original observation
    obs_a = np.copy(observation)
    laser_0 = n_agents + 1

    for a_idx in range(1,n_agents): # Change perspective of agents 1 to 3
        # Swap agent layer of agent a_idx
        obs_a[a_idx, [0, a_idx]] = observation[a_idx, [a_idx, 0]]
        # Store agent position for subsequent agents
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

    return flattened_obs+1 # +1 to have 0 as empty cells and no overlap with agent 0 identifier


def get_fixed_features(obs):
    """ Gets relevant data of the fixed features, including: wall coordinates, exit coordinates, initial gem coordinates, laser source coordinates (per agent) and laser beam lines (lists of coordinates) (per agent)."""
    A, L, H, W = obs.shape
    
    # Extract fixed global elements layers in shapes: (H,W)
    wall_mask = obs[0, A, :, :] == 1 # Wall layer after agent layers
    exit_mask = obs[0, -1, :, :] == 1 # last layer
    gem_mask = obs[0, -2, :, :] == 1 # second last layer
    # build global elements mask (same for all agents)
    walls_yx = np.argwhere(wall_mask)
    exits_yx = np.argwhere(exit_mask)
    gems_yx = np.argwhere(gem_mask) # gems for initial pos

    agent_l_sources = []
    agent_l_beams = []
    for i in range(A):
        layer = obs[i,A+1+i,:,:]
        sources_mask = layer == -1
        agent_l_sources.append(np.argwhere(sources_mask).tolist())
        visited = np.zeros_like(layer, dtype=bool)
        beams = {}
        for y in range(H):
            for x in range(W):
                if layer[y, x] != 1 or visited[y, x]:
                    continue
                # Decide orientation: look right vs down
                if x + 1 < W and layer[y, x + 1] == 1:  # horizontal beam
                    start = (x,y) # Inverted to table logic, used for end user
                    xs = []
                    cx = x
                    while cx < W and layer[y, cx] == 1:
                        visited[y, cx] = True
                        xs.append(cx)
                        cx += 1
                    coords = np.column_stack([np.full(len(xs), y, int), xs])
                    end = (cx,y)
                else:   # vertical beam
                    start = (x,y)
                    ys = []
                    cy = y
                    while cy < H and layer[cy, x] == 1:
                        visited[cy, x] = True
                        ys.append(cy)
                        cy += 1
                    coords = np.column_stack([ys, np.full(len(ys), x, int)])
                    end = (x,cy)
                beams[(start,end)] = coords
        agent_l_beams.append(beams)
    return (walls_yx.tolist(), exits_yx.tolist(), gems_yx.tolist(), agent_l_sources, agent_l_beams)


def dist_to_beam(ag_pos, beam, delta = False):
    """ Gives the distance by hypothenus or deltax and deltay between a given point (y,x) and a beam (list of points)
    """
    beam_arr = np.asarray(beam)
    ag_arr   = np.asarray(ag_pos)

    deltas = beam_arr - ag_arr
    dy_arr = deltas[:, 0]
    dx_arr = deltas[:, 1]

    # Use absolute value so that 0 is considered closer than -3
    idx_y = np.abs(dy_arr).argmin()
    idx_x = np.abs(dx_arr).argmin()

    # get the signed values
    dy = dy_arr[idx_y]
    dx = dx_arr[idx_x]

    if delta: return [dx.item(), dy.item()]
    else: return float(np.hypot(dy, dx))


def abstract_observation(obs, fix_feats, ag_pos):
    """
    Returns for each agent their abstracted observation. The shape of the abstracted observation might change from agent to agent."""
    ag_pos = ag_pos.tolist()

    A, L, H, W = obs.shape

    walls_yx, exits_yx, init_gems_yx, agent_l_sources, agent_l_beams = fix_feats

    # Extract dynamic global elements layers in shapes: (H,W)
    gem_mask = obs[0, -2, :, :] == 1 # second last layer
    # build global elements mask (same for all agents)
    gems_yx = np.argwhere(gem_mask)

    # Contains per agent for each beam the distance to all agents, in their order
    ags_to_beams = []     
    for agb_i in range(A):
        ags_to_beams.append([]) # new agentbeam dimension
        ag_beams = agent_l_beams[agb_i]
        for beam_i, beam_key in enumerate(ag_beams):
            beam = ag_beams[beam_key]
            ags_to_beams[agb_i].append([]) # New beam dimension for agent agb_i
            for ag_i in range(A):   # For each agent distance
                if ag_i != agb_i: ags_to_beams[agb_i][beam_i].append(dist_to_beam(ag_pos[ag_i],beam))
                else: ags_to_beams[agb_i][beam_i].append(dist_to_beam(ag_pos[ag_i],beam,True))

    features = []
    # Agent-wise feature extraction (lasers, other agents)
    for i in range(A):
        agent_y, agent_x = ag_pos[i]
        f = []

        # current agent absolute position normalized [0,1]
        f += [agent_x / (W-1), agent_y / (H-1)] # 1

        # relative position & euclidean distance to other agents
        for j in range(A):
            if j != i:
                y_j, x_j = ag_pos[j]
                dx, dy = x_j - agent_x, y_j - agent_y
                f += [dx, dy, math.hypot(dx, dy)] # n_agents-1: rel position and euclidian distance

        # relative position to every gem
        for (y_g, x_g) in init_gems_yx:
            if (y_g, x_g) in gems_yx: f += [x_g - agent_x, y_g - agent_y] # 1 per gem: relative pos to gem
            else: f += [0,0] # If gem already taken -> very big distance

        # laser sources (own colour only; distance to source)
        for y_s, x_s in agent_l_sources[i]:
            f += [x_s - agent_x, y_s - agent_y] # 1 per agent laser source

        # relative position to closest exit
        if len(exits_yx):
            dists = [math.hypot(x_e - agent_x, y_e - agent_y) for (y_e, x_e) in exits_yx]
            idx = np.argmin(dists)
            y_e, x_e = exits_yx[idx]
            dx_e, dy_e = x_e - agent_x, y_e - agent_y
            f += [dx_e, dy_e, dists[idx]]
        else:
            f += [0, 0, 0]  # 3: dx, dy and dist to exit

        # wall distance in four cardinal directions
        wall_up, wall_down, wall_left, wall_right = H,H,W,W
        for (y, x) in walls_yx:
            if x == agent_x:
                if y < agent_y:
                    temp = agent_y-y
                    if temp < wall_up or wall_up == 0: wall_up = temp
                else:
                    temp = y - agent_y
                    if temp < wall_down or wall_down == 0: wall_down = temp
            if y == agent_y:
                if x < agent_x:
                    temp = agent_x - x
                    if temp < wall_left or wall_left == 0: wall_left = temp
                else:
                    temp = x - agent_x
                    if temp < wall_right or wall_right == 0: wall_right = temp
        f += [wall_up, wall_down, wall_left, wall_right] # 4: wall dist N/E/S/W

        # For each own laser: deltax/y and smallest ally distance
        danger_and_block = [False,False]
        for ag_b_i in range(len(ags_to_beams)):
            if ag_b_i == i: # Own lasers
                for ag_beam in ags_to_beams[ag_b_i]:
                    closest = min(ag_beam[:i] + ag_beam[i+1:])
                    f += list(ag_beam[i]) + [closest] # 2 per beam of agent: relative agent pos + smallest ally distance
            else: # Any other laser
                for ag_beam in ags_to_beams[ag_b_i]: # Only consider 1 -> lose some precision but OK, else dynamic
                    if ag_beam[i] <= 1: 
                        danger_and_block[0] = True  # In danger
                        if math.hypot(ag_beam[ag_b_i][0],ag_beam[ag_b_i][1]) <= 1: danger_and_block[1] = True # ally can block
                        else: danger_and_block[1] = False # No one can block
        f += danger_and_block # 2: is threat, someone to block threat 


        features.append(f)
    return features   # shape (n_agents, undefined)

def feature_labels(n_agents, n_gems, laser_keys):
    """ Compute labels for features extracted in abstract_observation for one agent!
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
        labels += [f"Delta x to gem {g+1}",
                f"Delta y to gem {g+1}"]

    # Relative distance to laser sources
    for l in range(len(laser_keys)):
        labels += [f"Delta x to own laser {l+1} source",
                f"Delta y to own laser {l+1} source"]

    # Relative distance to closest exit
    labels += ["Delta x to closest exit",
            "Delta y to closest exit",
            "Distance to closest exit"]

    # wall distances (cardinal)
    labels += ["Closest wall up", "Closest wall down", "Closest wall left", "Closest wall right"]

    for i,coords in enumerate(laser_keys):
        labels += [f"Delta x to laserbeam {i} {coords}",
                    f"Delta y to laserbeam {i} {coords}",
                    f"Smallest ally distance to laserbeam {i} {coords}"]
    labels += ["Threatened by laser (d<=1)?","Blocker in range of threat (d<=1)?"]
    return labels