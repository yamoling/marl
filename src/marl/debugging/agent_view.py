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
from marl.qlearning import QLearning
from marl import RLAlgorithm


@dataclass
class TrainingState:
    algo: QLearning
    current_epidode: EpisodeBuilder
    obs: Observation
    episode_num: int
    step_num: int

    def __init__(self, algo: QLearning) -> None:
        self.algo = algo
        self.current_epidode = EpisodeBuilder()
        self.obs = self.algo.env.reset()
        self.episode_num = 0
        self.step_num = 0

    def render(self):
        self.algo.env.render()

    def step(self) -> Transition:
        if self.current_epidode.is_done:
            episode = self.current_epidode.build()
            self.algo.after_episode(self.episode_num, episode)
            self.algo.logger.log("Train", episode.metrics, self.step_num)
            self.episode_num += 1
            self.algo.before_episode(self.episode_num)
            self.current_episode = EpisodeBuilder()
            self.obs = self.algo.env.reset()
        action = self.algo.choose_action(self.obs)
        obs_, reward, done, info = self.algo.env.step(action)
        transition = Transition(self.obs, action, reward, done, info, obs_)
        self.algo.after_step(self.step_num, transition)
        self.current_epidode.add(transition)
        self.obs = obs_
        self.step_num += 1


class QLearningInspector(RLAlgorithm):
    def __init__(self, algo: QLearning, webview=False, debug=False) -> None:
        super().__init__(algo.env, algo.test_env, "debug")
        self.algo = algo
        if webview:
            t = threading.Thread(target=lambda: asyncio.run(self.run_server()))
            t.daemon = True
            t.start()
            self.start_webview()
        else:
            asyncio.run(self.run_server())

    @staticmethod
    def start_webview():
        webview.create_window('Hello world', "http://localhost:5173")
        webview.start(debug=True)

    async def run_server(self):
        async with serve(self.handler, "localhost", 5172):
            await asyncio.Future()

    async def send_update(self, ws: WebSocketServerProtocol, training: TrainingState):
        qvalues: np.ndarray = self.algo.compute_qvalues(training.obs).detach().cpu().numpy()
        rgb_array = training.algo.env.render("rgb_array")
        await ws.send(json.dumps({
            "qvalues": qvalues.tolist(),
            "observations": training.obs.data.tolist(),
            "state": training.obs.state.tolist(),
            "extras": training.obs.extras.tolist(),
            "available": training.obs.available_actions.tolist(),
            "b64_rendering": base64.b64encode(cv2.imencode(".jpg", rgb_array)[1]).decode("ascii")
        }))

    async def handler(self, ws: WebSocketServerProtocol, path):
        training = TrainingState(self.algo)
        training.render()
        await self.send_update(ws, training)
        async for message in ws:
            data = json.loads(message)
            print("New message from websocket: ", data)
            match data["command"]:
                case "step": steps = 1
                case "skip": steps = data["amount"]
                case other: print(f"Unknwon command {other}")
            for _ in range(steps):
                training.step()
            training.render()
            await self.send_update(ws, training)


    def choose_action(self, observation) -> np.ndarray[np.int64]:
        return self.algo.choose_action(observation)


