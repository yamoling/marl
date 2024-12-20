import psutil
from psutil import Process
from marl.models import Experiment
from threading import Thread
import asyncio
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed


class WebsocketServer(Thread):
    def __init__(self, port: int):
        super().__init__(daemon=True)
        self.port = port
        self.clients = list()

    async def _accept_connection(self, websocket):
        print("new client connected")
        self.clients.append(websocket)

    async def main(self):
        print("starting watcher websocket server")
        async with serve(self._accept_connection, "", self.port):
            await asyncio.Future()

    def run(self):
        asyncio.run(self.main())

    def send_message(self, message: str):
        to_remove = []
        for client in self.clients:
            try:
                asyncio.run(client.send(message))
            except ConnectionClosed:
                to_remove.append(client)
        for client in to_remove:
            self.clients.remove(client)


class Watcher(Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.pids = dict[int, str]()
        self.experiments = dict[str, list[int]]()
        self.server = WebsocketServer(5002)

    def watch(self, logdir: str):
        self.experiments[logdir] = []
        exp = Experiment.load(logdir)
        for run in exp.runs:
            pid = run.get_pid()
            if pid is not None:
                process = Process(pid)
                if process.status() == psutil.STATUS_RUNNING:
                    self.pids[pid] = logdir
                    self.experiments[logdir].append(process.pid)
        if len(self.experiments[logdir]) == 0:
            self.experiments.pop(logdir)
            self.notify_finished(logdir)

    def start(self):
        self.server.start()
        return super().start()

    def notify_finished(self, logdir: str):
        print(f"Experiment {logdir} has finished, sending message to clients")
        self.server.send_message(logdir)

    def _on_process_terminated(self, process: Process):
        print(f"Process {process.pid} has terminated")
        logdir = self.pids.pop(process.pid)
        self.experiments[logdir].remove(process.pid)
        if len(self.experiments[logdir]) == 0:
            self.experiments.pop(logdir)
            self.notify_finished(logdir)

    def run(self):
        while True:
            processes = list(Process(pid) for pid in self.pids.keys())
            gone, alive = psutil.wait_procs(processes, timeout=5, callback=self._on_process_terminated)
            for process in alive:
                if process.status() == psutil.STATUS_ZOMBIE:
                    self._on_process_terminated(process)


def run(logdir: str = "logs/"):
    import os

    watcher = Watcher()
    for directory in os.listdir(logdir):
        directory = os.path.join(logdir, directory)
        if Experiment.is_experiment_directory(directory):
            watcher.watch(directory)

    watcher.start()
