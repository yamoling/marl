import threading
import json
import os
import asyncio
from dataclasses import dataclass
from websockets.server import serve, WebSocketServerProtocol
from rlenv.models import Metrics
from marl.utils.others import get_available_port
from marl.models import ReplayEpisodeSummary
from .logger_interface import Logger


@dataclass
class LogItem:
    tag: str
    metrics: Metrics | None
    time_step: int

    def to_json(self):
        data = {
            "tag": self.tag,
            "step": self.time_step
        } 
        if self.metrics is not None:
            data["metrics"] = self.metrics.to_json()
        return data

@dataclass
class WSLogger(Logger):
    """A logger that logs to a websocket"""
    clients: set[WebSocketServerProtocol]
    port: int

    def __init__(self, logdir: str, port: int=None) -> None:
        super().__init__(logdir)
        if port == None:
            port = get_available_port()
        self.port = port
        self.clients = set()
        self.messages = asyncio.Queue()
        self._stop = False
        self._started = False
        self.start()

    def stop(self):
        self._stop = True

    def start(self):
        self._started = False
        threading.Thread(target=lambda: asyncio.run(self.run()), daemon=True).start()
        while not self._started:
            pass

    async def connection_handler(self, ws: WebSocketServerProtocol, path):
        print(f"New connection at {path}")
        self.clients.add(ws)
        # The client gets dsicconnected when this method returns
        while not self._stop:
            await asyncio.sleep(0.25)

    async def run(self):
        await asyncio.gather(self.accept_connections(), self.update_loop())

    async def update_loop(self):
        while not self._stop:
            log: ReplayEpisodeSummary = await self.messages.get()
            json_data = json.dumps(log.to_json())
            to_remove = set()
            for client in self.clients:
                try:
                    await client.send(json_data)
                except:
                    to_remove.add(client)
            self.clients.difference_update(to_remove)

    async def accept_connections(self):
        print("Starting websocket server on port", self.port, "...")
        try:
            async with serve(self.connection_handler, "0.0.0.0", self.port):
                print("Websocket server started")
                self._started = True
                while not self._stop:
                    await asyncio.sleep(1)
        except OSError:
            print("Could not start websocket server because the port is already in use")
            # TODO: handle this correctly
            exit(1)
        print("Websocket server stopped")

    def log(self, tag: str, data: Metrics, time_step: int):
        directory = os.path.join(self.logdir, tag, f"{time_step}")
        # self.messages.put_nowait(LogItem(tag, data, time_step))
        self.messages.put_nowait(ReplayEpisodeSummary(directory, data))

    def print(self, tag: str, data):
        pass
        
    def flush(self, prefix: str | None = None):
        pass