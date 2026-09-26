"""
Live events (Server-Sent Events) with one poller shared by all clients.

The poller runs only while at least one client is connected. Every `running_interval` it re-reads
the experiments with running runs and emits `run-progress` when a run's status or latest step
changes (including a final message when a run stops running). Every `scan_interval` it scans the
logs root and emits `experiment-added`, `experiment-removed` and `experiment-changed`
(experiment.json / run.json / run set changes).
"""

import asyncio
import logging
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson

from .. import settings
from ..data.cache import file_key
from ..data.library import Library
from ..data.records import EXPERIMENT_FILE, RUN_FILE, ExperimentRecord, run_dirs

logger = logging.getLogger(__name__)

Event = tuple[str, dict[str, Any]]


@dataclass(frozen=True)
class EventsConfig:
    running_interval: float = settings.EVENTS_RUNNING_INTERVAL_S
    scan_interval: float = settings.EVENTS_SCAN_INTERVAL_S
    ping_interval: float = settings.EVENTS_PING_INTERVAL_S


def sse(event: str, data: Any) -> bytes:
    """@ai-generated"""
    return b"event: " + event.encode() + b"\ndata: " + orjson.dumps(data) + b"\n\n"


def meta_key(path: Path) -> tuple:
    """Stats of experiment.json and of each run.json: changes trigger `experiment-changed`. @ai-generated"""
    return (file_key(path / EXPERIMENT_FILE), tuple((d.name, file_key(d / RUN_FILE)) for d in run_dirs(path)))


def _progress(record: ExperimentRecord, run_id: str) -> dict[str, Any] | None:
    """@ai-generated"""
    run = record.run(run_id)
    if run is None:
        return None
    return {"experiment": record.id, "run": run.id, "status": run.status, "progress": run.progress, "latest_step": run.latest_step}


class EventHub:
    def __init__(self, library: Library, config: EventsConfig | None = None):
        self.library = library
        self.config = config or EventsConfig()
        self._lock = threading.Lock()
        self._subscribers = set[asyncio.Queue]()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None
        self._known: dict[str, tuple] | None = None
        """Meta keys of the known experiments; None before the first scan."""
        self._running = dict[str, dict[str, Any]]()
        """Last `run-progress` payload of each running run."""
        self._watched = set[str]()
        self._last_scan = float("-inf")

    # ------------------------------------------------------------ Subscribers

    @property
    def n_subscribers(self) -> int:
        return len(self._subscribers)

    @property
    def initialized(self) -> bool:
        return self._known is not None

    async def subscribe(self) -> asyncio.Queue:
        """Register a client and start the poller if needed. @ai-generated"""
        queue = asyncio.Queue[Event | None]()
        loop = asyncio.get_running_loop()
        self._subscribers.add(queue)
        if self._task is None or self._task.done() or self._loop is not loop:
            self._loop = loop
            self._task = loop.create_task(self._run(), name="studio-events-poller")
        return queue

    def unsubscribe(self, queue: asyncio.Queue):
        """Unregister a client; the poller stops with the last one. @ai-generated"""
        self._subscribers.discard(queue)
        if not self._subscribers and self._task is not None:
            self._task.cancel()
            self._task = None

    def publish(self, event: str, data: dict[str, Any]):
        """Must be called from the event loop. @ai-generated"""
        for queue in list(self._subscribers):
            queue.put_nowait((event, data))

    def publish_threadsafe(self, event: str, data: dict[str, Any]):
        """Send a launcher event from its background waiter to active SSE clients. @ai-generated"""
        loop = self._loop
        if loop is not None and not loop.is_closed() and self._subscribers:
            loop.call_soon_threadsafe(self.publish, event, data)

    def notify_changed(self, experiment_id: str):
        """
        Thread-safe hint (e.g. after a launch): watch the experiment's runs at the next tick and emit
        `experiment-changed` for it.

        @ai-generated
        """
        with self._lock:
            self._watched.add(experiment_id)
        loop = self._loop
        if loop is not None and not loop.is_closed() and self._subscribers:
            loop.call_soon_threadsafe(self.publish, "experiment-changed", {"experiment": experiment_id})

    def close(self):
        """Thread-safe: end every open stream and stop the poller. @ai-generated"""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._close_now)
        except RuntimeError:
            pass  # The loop has just been closed

    def _close_now(self):
        """@ai-generated"""
        for queue in list(self._subscribers):
            queue.put_nowait(None)
        if self._task is not None:
            self._task.cancel()
            self._task = None

    # ------------------------------------------------------------ Polling

    async def snapshot(self) -> dict[str, Any]:
        """`{running: [...]}`, scanning the logs root first if the poller has not done it yet. @ai-generated"""
        if not self.initialized:
            await asyncio.to_thread(self._scan)
        with self._lock:
            return {"running": list(self._running.values())}

    async def _run(self):
        """@ai-generated"""
        try:
            while True:
                scan = time.monotonic() - self._last_scan >= self.config.scan_interval
                try:
                    events = await asyncio.to_thread(self._scan if scan else self._poll)
                except Exception:
                    logger.exception("Events poller failed")
                    events = []
                for event in events:
                    self.publish(*event)
                await asyncio.sleep(self.config.running_interval)
        except asyncio.CancelledError:
            pass

    def _progress_events(self, records: dict[str, ExperimentRecord], polled: set[str]) -> list[Event]:
        """Compare the running runs of `polled` experiments with the previous state. @ai-generated"""
        events = list[Event]()
        with self._lock:
            previous = dict(self._running)
            for exp_id in polled:
                record = records.get(exp_id)
                current_ids = {r.id for r in record.runs if r.status == "RUNNING"} if record else set()
                before_ids = {run_id for run_id, p in previous.items() if p["experiment"] == exp_id}
                for run_id in current_ids | before_ids:
                    payload = _progress(record, run_id) if record else None
                    if payload is None:  # The run or experiment disappeared
                        payload = previous[run_id] | {"status": "UNKNOWN"}
                    old = previous.get(run_id)
                    if old is None or (old["status"], old["latest_step"]) != (payload["status"], payload["latest_step"]):
                        events.append(("run-progress", payload))
                    if payload["status"] == "RUNNING":
                        self._running[run_id] = payload
                    else:
                        self._running.pop(run_id, None)
                if record is None or record.running_runs == 0:
                    self._watched.discard(exp_id)
                else:
                    self._watched.add(exp_id)
        return events

    def _poll(self) -> list[Event]:
        """Re-read the experiments with running runs. @ai-generated"""
        with self._lock:
            polled = set(self._watched) | {p["experiment"] for p in self._running.values()}
        records = {exp_id: r for exp_id in polled if (r := self.library.get(exp_id)) is not None}
        return self._progress_events(records, polled)

    def _scan(self) -> list[Event]:
        """Scan the logs root: added/removed/changed experiments, then running runs. @ai-generated"""
        self._last_scan = time.monotonic()
        ids = self.library.ids()
        keys = {exp_id: meta_key(self.library.root / exp_id) for exp_id in ids}
        events = list[Event]()
        first = self._known is None
        known = self._known or {}
        if not first:
            events += [("experiment-added", {"experiment": i}) for i in ids if i not in known]
            events += [("experiment-removed", {"experiment": i}) for i in known if i not in keys]
            events += [("experiment-changed", {"experiment": i}) for i in ids if i in known and known[i] != keys[i]]
        records = {exp_id: r for exp_id in ids if (r := self.library.get(exp_id)) is not None}
        with self._lock:
            polled = set(records) | set(self._watched) | {p["experiment"] for p in self._running.values()}
        progress = self._progress_events(records, polled)
        self._known = keys
        return events + ([] if first else progress)


async def stream(hub: EventHub) -> AsyncIterator[bytes]:
    """SSE stream of one client: a snapshot, then events, with a ping when idle. @ai-generated"""
    queue = await hub.subscribe()
    try:
        yield sse("snapshot", await hub.snapshot())
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=hub.config.ping_interval)
            except TimeoutError:
                yield sse("ping", {})
                continue
            if item is None:
                return
            yield sse(*item)
    finally:
        hub.unsubscribe(queue)
