import asyncio
import threading
import json
from websockets.server import serve, WebSocketServerProtocol
from rlenv.models import EpisodeBuilder, Episode, Transition

import marl
from marl.models import ReplayMemory
from marl.utils.others import encode_b64_image

from .debug_wrapper import QLearningDebugger


class DebugRunner(marl.Runner):
    def __init__(self, env, test_env, algo: QLearningDebugger, logdir: str, memory: ReplayMemory = None):
        self._q_function = algo.algo
        super().__init__(env, test_env=test_env, algo=algo, logdir=logdir)
        self.step_num = 0
        self.episode_num = 0
        self.current_episode = EpisodeBuilder()
        self.obs = self._env.reset()
        self._algo.before_episode(self.episode_num)
        self.memory = memory
        self._prev_frame = None
        self._current_frame = None
        self.stop = False
        # Type hinting
        self._algo: QLearningDebugger
        # Start server
        self.start_ws_server()
        self.write_experiment_summary()
        
    def start_ws_server(self):
        t = threading.Thread(target=lambda: asyncio.run(self.run_server()))
        t.daemon = True
        t.start()
    
    async def send_update(self, episode: Episode, ws: WebSocketServerProtocol):
        data =  {
            "step": self.step_num,
            "episode": self.episode_num,
            "metrics": episode.metrics.to_json()
        }
        await ws.send(json.dumps(data))

    async def client_handler(self, ws: WebSocketServerProtocol, _path):
        print("Client connected")
        json_data = json.loads(await ws.recv())
        await self.train(json_data["steps"], ws)

    async def run_server(self):
        while not self.stop:
            try:
                async with serve(self.client_handler, "0.0.0.0", 5172):
                    while not self.stop:
                        await asyncio.sleep(0.25)
            # If the previous websocket server is still running
            except OSError:
                print("Could not start websocket server, retrying in 0.25s")
                await asyncio.sleep(0.25)
        print("Websocket server closed")

    async def train(self, n_steps: int, ws: WebSocketServerProtocol):
        stop = self.step_num + n_steps
        while self.step_num < stop:
            if self.current_episode.is_done:
                # Finish episode
                self._qvalues = []
                episode = self.current_episode.build()
                self._algo.after_episode(self.episode_num, episode)
                self._logger.log("Train", episode.metrics, self.step_num)
                await self.send_update(episode, ws)
                # Start new episode
                self.episode_num += 1
                self.current_episode = EpisodeBuilder()
                self._algo.before_episode(self.episode_num)
                self.obs = self._env.reset()
            action = self._algo.choose_action(self.obs)
            obs_, reward, done, info = self._env.step(action)
            transition = Transition(self.obs, action, reward, done, info, obs_)
            self._algo.after_step(transition, self.step_num)
            self.current_episode.add(transition)
            self.obs = obs_
            self.step_num += 1
            self._prev_frame = self._current_frame
            self._current_frame = self._env.render("rgb_array")

    def get_state(self) -> dict:
        episode = self.current_episode.build()
        episode_json = episode.to_json()
        episode_json["qvalues"] = self._algo.training_qvalues + self._q_function.compute_qvalues(self.obs).tolist()
        episode_json["available_actions"] += self._env.get_avail_actions().tolist()
        episode_json["obs"] += self.obs.data.tolist()
        episode_json["extras"] += self.obs.extras.tolist()
        return {
            "episode": episode_json,
            "prev_frame": encode_b64_image(self._prev_frame),
            "current_frame": encode_b64_image(self._current_frame)
        }
