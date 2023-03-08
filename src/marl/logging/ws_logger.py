import threading
import json
import asyncio
from dataclasses import dataclass
from websockets.server import serve, WebSocketServerProtocol
from rlenv.models import Metrics
from marl.utils.others import get_available_port
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
    clients: list[WebSocketServerProtocol]
    port: int
    _disconnect_clients: bool

    def __init__(self, logdir: str, port: int=None) -> None:
        super().__init__(logdir)
        if port == None:
            port = get_available_port()
        self.port = port
        self.clients = []
        self.messages = asyncio.Queue()
        self._start_background_tasks()
        self._disconnect_clients = False

    def _start_background_tasks(self):
        t = threading.Thread(target=lambda: asyncio.run(self.run()))
        t.daemon = True
        t.start()

    async def connection_handler(self, ws: WebSocketServerProtocol, path):
        print(f"New connection at {path}")
        self._disconnect_clients = False
        self.clients.append(ws)
        # The client gets dsicconnected when this method returns
        while not self._disconnect_clients:
            await asyncio.sleep(0.25)

    async def run(self):
        await asyncio.gather(self.update_loop(), self.accept_connections())

    async def update_loop(self):
        while True:
            log: LogItem = await self.messages.get()
            json_data = json.dumps(log.to_json())
            to_remove = set()
            for client in self.clients:
                try:
                    await client.send(json_data)
                except:
                    to_remove.add(client)
            for client in to_remove:
                self.clients.remove(client)

    async def accept_connections(self):
        try:
            async with serve(self.connection_handler, "0.0.0.0", self.port):
                while True:
                    await asyncio.sleep(1)
        except OSError:
            print("Could not start websocket server because the port is already in use")

    def log(self, tag: str, data: Metrics, time_step: int):
        if tag == "done":
            self._disconnect_clients = True
        else:
            self.messages.put_nowait(LogItem(tag, data, time_step))

    def print(self, tag: str, data):
        pass
        
    def flush(self, prefix: str | None = None):
        pass