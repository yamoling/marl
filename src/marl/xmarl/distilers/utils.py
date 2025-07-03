import numpy as np
from marl.models import Experiment

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
    
def abstract_observation(self, observation, extras):
    """Abstracts a given observation to high-level components"""
    pass
