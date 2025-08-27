import threading
import os
import asyncio
from typing import ClassVar
from websockets.exceptions import ConnectionClosed
from marl.models.replay_episode import LightEpisodeSummary
import orjson

from websockets.asyncio.server import serve, ServerConnection


class WSLogger:
    """A logger that logs to a websocket"""

    WS_FILE: ClassVar[str] = "ws_port"

    clients: set[ServerConnection]
    port: int

    def __init__(self, port: int) -> None:
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

    async def connection_handler(self, ws: ServerConnection):
        self.clients.add(ws)
        # The client gets disconnected when this method returns
        while not self._stop:
            await asyncio.sleep(0.25)

    async def run(self):
        await asyncio.gather(self.accept_connections(), self.update_loop())

    async def update_loop(self):
        while not self._stop:
            log: LightEpisodeSummary = await self.messages.get()
            data = orjson.dumps(log)
            to_remove = set()
            for client in self.clients:
                try:
                    await client.send(data)
                except ConnectionClosed:
                    to_remove.add(client)
            self.clients.difference_update(to_remove)

    async def accept_connections(self):
        try:
            # If self.port is None, then the OS will choose a random port
            async with serve(self.connection_handler, "0.0.0.0", self.port) as server:
                # Retrieve the port number that was chosen
                self.port: int = list(server.sockets)[0].getsockname()[1]
                print("Starting websocket server on port", self.port, "...")
                self._started = True
                while not self._stop:
                    await asyncio.sleep(1)
        except OSError:
            print("Could not start websocket server because the port is already in use...")
            exit(1)

    def log(self, tag: str, data: dict[str, float], time_step: int):
        directory = os.path.join(tag, f"{time_step}")
        self.messages.put_nowait(LightEpisodeSummary(directory, data))

    def __del__(self):
        self.stop()

    def close(self):
        self.stop()
