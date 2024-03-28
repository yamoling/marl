import numpy as np
import itertools
from rlenv import RLEnv, DiscreteActionSpace, Observation


N_ROWS = 11
N_COLS = 12


class CoordinatedGrid(RLEnv):
    """
    Coordinated grid world environment used in the EMC paper to test the effectiveness of the proposed method.
    https://proceedings.neurips.cc/paper_files/paper/2021/file/1e8ca836c962598551882e689265c1c5-Paper.pdf
    """

    def __init__(
        self,
        episode_limit=30,
        time_penalty=2,
    ):
        super().__init__(
            action_space=DiscreteActionSpace(2, 5, ["SOUTH", "NORTH", "WEST", "EAST", "STAY"]),
            observation_shape=(N_ROWS + N_COLS,),
            state_shape=(N_ROWS + N_COLS,) * 2,
        )
        self._episode_steps = 0
        self.episode_limit = episode_limit
        self.center = N_COLS // 2
        ###larger gridworld
        visible_row = [i for i in range(N_ROWS // 2 - 2, N_ROWS // 2 + 3)]
        visible_col = [i for i in range(N_COLS // 2 - 3, N_COLS // 2 + 3)]
        self.vision_index = [[i, j] for i, j in list(itertools.product(visible_row, visible_col))]
        self.agents_location = [[0, 0], [N_ROWS - 1, N_COLS - 1]]
        self.time_penalty = time_penalty

    def reset(self):
        self.agents_location = [[0, 0], [N_ROWS - 1, N_COLS - 1]]
        self._episode_steps = 0
        return self.observation()

    def get_state(self):
        return self.observation().state

    def observation(self):
        obs_1 = [[0 for _ in range(N_ROWS)], [0 for _ in range(N_COLS)]]
        # obs_2 = obs_1.copy()
        import copy

        obs_2 = copy.deepcopy(obs_1)

        obs_1[0][self.agents_location[0][0]] = 1
        obs_1[1][self.agents_location[0][1]] = 1
        obs_1 = obs_1[0] + obs_1[1]

        obs_2[0][self.agents_location[1][0]] = 1
        obs_2[1][self.agents_location[1][1]] = 1
        obs_2 = obs_2[0] + obs_2[1]

        if self.agents_location[0] in self.vision_index and self.agents_location[1] in self.vision_index:
            temp = obs_1.copy()
            obs_1 += obs_2.copy()
            obs_2 += temp.copy()
        elif self.agents_location[0] in self.vision_index:
            obs_2 += obs_1.copy()
            obs_1 += [0 for _ in range(N_ROWS + N_COLS)]
        elif self.agents_location[1] in self.vision_index:
            obs_1 += obs_2.copy()
            obs_2 += [0 for _ in range(N_ROWS + N_COLS)]
        else:
            obs_2 += [0 for _ in range(N_ROWS + N_COLS)]
            obs_1 += [0 for _ in range(N_ROWS + N_COLS)]

        state = np.array(obs_1 + obs_2)
        obs_data = np.array([obs_1, obs_2])
        return Observation(obs_data, self.available_actions(), state)

    def available_actions(self):
        avail_actions = np.ones((self.n_agents, self.n_actions))
        for agent_num, (y, x) in enumerate(self.agents_location):
            if x == 0:
                avail_actions[agent_num, 0] = 0
            elif x == N_ROWS - 1:
                avail_actions[agent_num, 1] = 0
            if y == 0:
                avail_actions[agent_num, 2] = 0
            # Check for center line (depends on the agent number)
            elif y == self.center + agent_num - 1:
                avail_actions[agent_num, 3] = 0
        return avail_actions.tolist()

    def step(self, actions):
        # print('State is {}, action is {}'.format(self.state, actions))
        for idx, action in enumerate(actions):
            match action:
                case 0:
                    self.agents_location[idx][0] -= 1
                case 1:
                    self.agents_location[idx][0] += 1
                case 2:
                    self.agents_location[idx][1] -= 1
                case 3:
                    self.agents_location[idx][1] += 1
                case 4:
                    pass
                case _:
                    raise ValueError(f"Invalid action {action} for agent {idx}!")

        self._episode_steps += 1

        # print('Next state is {}'.format(self.state))
        terminated = self._episode_steps >= self.episode_limit
        env_info = {"battle_won": False}
        n_arrived = self.n_agents_arrived()
        if n_arrived == 1:
            reward = -self.time_penalty
        elif n_arrived == 2:
            reward = 10
            terminated = True
            env_info = {"battle_won": True}
        else:
            reward = 0
        return self.observation(), reward, terminated, terminated, env_info

    def n_agents_arrived(self):
        n = 0
        if self.agents_location[0] == [N_ROWS // 2, self.center - 1]:
            n += 1
        if self.agents_location[1] == [N_ROWS // 2, self.center]:
            n += 1
        return n

    def render(self, *_):
        print("Agents location: ", self.agents_location)
        for row in range(N_ROWS):
            for col in range(N_COLS):
                if [row, col] in self.agents_location:
                    print("X", end=" ")
                else:
                    print(".", end=" ")
            print()
