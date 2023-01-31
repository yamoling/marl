import asyncio
import threading
import json
import base64
import numpy as np
import webview
import cv2
from dataclasses import dataclass
from websockets.server import serve, WebSocketServerProtocol
from rlenv.models import EpisodeBuilder, Transition, Observation
from marl.qlearning import DeepQLearning
from marl import RLAlgorithm


@dataclass
class TrainingState:
    algo: DeepQLearning
    current_episode: EpisodeBuilder
    last_reward: float
    obs: Observation
    episode_num: int
    step_num: int
    

    def __init__(self, algo: DeepQLearning) -> None:
        self.algo = algo
        self.current_episode = EpisodeBuilder()
        self.obs = self.algo.env.reset()
        self.episode_num = 0
        self.step_num = 0
        self.last_reward = None
        self.algo.before_episode(self.episode_num)

    def step(self):
        print("Step", self.step_num)
        if self.current_episode.is_done:
            episode = self.current_episode.build()
            self.algo.after_episode(self.episode_num, episode)
            self.algo.logger.log("Train", episode.metrics, self.step_num)
            self.episode_num += 1
            self.algo.before_episode(self.episode_num)
            self.current_episode = EpisodeBuilder()
            self.obs = self.algo.env.reset()
            self.last_reward = None
        else:
            action = self.algo.choose_action(self.obs)
            obs_, self.last_reward, done, info = self.algo.env.step(action)
            transition = Transition(self.obs, action, self.last_reward, done, info, obs_)
            self.algo.after_step(self.step_num, transition)
            self.current_episode.add(transition)
            self.obs = obs_
            self.step_num += 1


class QLearningInspector(RLAlgorithm):
    def __init__(self, algo: DeepQLearning) -> None:
        super().__init__(algo.env, algo.test_env, "debug")
        self.algo = algo
        
    def run(self, webview=False, debug=False):
        if webview:
            t = threading.Thread(target=lambda: asyncio.run(self.run_server()))
            t.daemon = True
            t.start()
            self.start_webview(debug)
        else:
            asyncio.run(self.run_server())

    @staticmethod
    def start_webview(debug: bool):
        webview.create_window('Hello world', "http://localhost:5173")
        webview.start(debug=debug)

    async def run_server(self):
        # Strange stuff that needs to be done otherwise the ws server stops
        async with serve(self.handler, "localhost", 5172):
            await asyncio.Future()

    async def send_update(self, ws: WebSocketServerProtocol, training: TrainingState):
        qvalues: np.ndarray = self.algo.compute_qvalues(training.obs).detach().cpu().numpy()
        rgb_array = training.algo.env.render("rgb_array")
        done = training.current_episode.is_done
        await ws.send(json.dumps({
            "qvalues": qvalues.tolist(),
            "observations": training.obs.data.tolist(),
            "state": training.obs.state.tolist(),
            "extras": training.obs.extras.tolist(),
            "done": done,
            "reward": training.last_reward,
            "available": training.obs.available_actions.tolist(),
            "b64_rendering": base64.b64encode(cv2.imencode(".jpg", rgb_array)[1]).decode("ascii"),
        }))

    async def handler(self, ws: WebSocketServerProtocol, _path):
        training = TrainingState(self.algo)
        self.algo.seed(0)
        await self.send_update(ws, training)
        async for message in ws:
            print(message)
            data: dict = json.loads(message)
            steps = data.get("amount", 1)
            updateUI = data.get("command") != "skip"
            for _ in range(steps - 1):
                training.step()
                if updateUI:
                    await self.send_update(ws, training)
            training.step()
            await self.send_update(ws, training)

    def choose_action(self, observation) -> np.ndarray[np.int64]:
        return self.algo.choose_action(observation)
