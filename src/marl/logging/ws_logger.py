import threading
import json
import os
import asyncio
from dataclasses import dataclass
from typing import ClassVar
from websockets.server import serve, WebSocketServerProtocol
from rlenv.models import Metrics
from marl.models.replay_episode import ReplayEpisodeSummary
from .logger_interface import Logger



@dataclass
class WSLogger(Logger):
    """A logger that logs to a websocket"""
    WS_FILE: ClassVar[str] = "ws_port"

    clients: set[WebSocketServerProtocol]
    port: int

    def __init__(self, logdir: str, port: int=None) -> None:
        super().__init__(logdir)
        self.port = port
        self.clients = set()
        self.messages = asyncio.Queue()
        self._stop = False
        self._started = False
        self._ws_file = os.path.join(logdir, WSLogger.WS_FILE)
        self.start()

    def stop(self):
        self._stop = True
        try:
            os.remove(self._ws_file)
        except FileNotFoundError:
            pass

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
            data = json.dumps(log.to_json())
            to_remove = set()
            for client in self.clients:
                try:
                    await client.send(data)
                except:
                    to_remove.add(client)
            self.clients.difference_update(to_remove)

    async def accept_connections(self):
        try:
            # If self.port is None, then the OS will choose a random port
            async with serve(self.connection_handler, "0.0.0.0", self.port) as s:
                # Retrieve the port number that was chosen
                self.port: int = s.sockets[0].getsockname()[1]
                print("Starting websocket server on port", self.port, "...")
                with open(self._ws_file, "w") as f:
                    f.write(str(self.port))
                self._started = True
                while not self._stop:
                    await asyncio.sleep(1)
        except OSError:
            print("Could not start websocket server because the port is already in use...")
            exit(1)
            

    def log(self, tag: str, data: Metrics, time_step: int):
        directory = os.path.join(self.logdir, tag, f"{time_step}")
        self.messages.put_nowait(ReplayEpisodeSummary(directory, data))

    def print(self, tag: str, data):
        pass
        
    def flush(self, prefix: str | None = None):
        pass

    def __del__(self):
        self.stop()
