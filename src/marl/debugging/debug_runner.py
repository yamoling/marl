import asyncio
import threading
import json
from websockets.server import serve, WebSocketServerProtocol
import base64
import cv2
from rlenv.models import EpisodeBuilder, Episode, Transition
import marl
from marl.models import ReplayMemory
from marl.qlearning import IDeepQLearning

from .file_wrapper import FileWrapper


class DebugRunner(marl.Runner):
    def __init__(self, env, test_env, algo: IDeepQLearning, logdir: str, memory: ReplayMemory = None):
        self._q_function = algo
        if not isinstance(algo, FileWrapper):
            algo = FileWrapper(algo, logdir)
        super().__init__(env, test_env=test_env, algo=algo, logdir=logdir)
        self.step_num = 0
        self.episode_num = 0
        self.current_episode = EpisodeBuilder()
        self.obs = self._env.reset()
        self._algo.before_episode(self.episode_num)
        self.memory = memory
        self._prev_frame = None
        self._current_frame = None
        # Type hinting
        self._algo: FileWrapper
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
        # Strange stuff that needs to be done otherwise the ws server stops
        async with serve(self.client_handler, "0.0.0.0", 5172):
            await asyncio.Future()

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

    def encode_frame(self, frame):
        if frame is None:
            return ""
        return base64.b64encode(cv2.imencode(".jpg", frame)[1]).decode("ascii")

    def get_state(self) -> dict:
        episode = self.current_episode.build()
        episode_json = episode.to_json()
        episode_json["qvalues"] = self._algo.training_qvalues + self._q_function.compute_qvalues(self.obs).tolist()
        episode_json["available_actions"] += self._env.get_avail_actions().tolist()
        episode_json["obs"] += self.obs.data.tolist()
        episode_json["extras"] += self.obs.extras.tolist()
        return {
            "episode": episode_json,
            "prev_frame": self.encode_frame(self._prev_frame),
            "current_frame": self.encode_frame(self._current_frame)
        }
