"""A tiny in-process event bus for server-sent events.

Subscribers get their own bounded queue. A slow browser tab must never grow memory
without limit, so when a queue is full the oldest event is dropped.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from typing import AsyncIterator


class EventBus:
    def __init__(self, max_queue: int = 200) -> None:
        self._lock = threading.Lock()
        self._subs: list[queue.Queue] = []
        self._max = max_queue

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=self._max)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            try:
                self._subs.remove(q)
            except ValueError:
                pass

    def publish(self, kind: str, data: dict | None = None) -> None:
        payload = {"type": kind, "ts": int(time.time() * 1000), "data": data or {}}
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(payload)
            except queue.Full:
                try:
                    q.get_nowait()
                    q.put_nowait(payload)
                except (queue.Empty, queue.Full):
                    pass

    async def stream(self, q: queue.Queue) -> AsyncIterator[str]:
        """Yield SSE frames until the client disconnects.

        ``queue.Queue.get`` is blocking, so the timeout wait runs in a worker thread.
        Doing it inline would freeze the server's event loop for every idle client.
        """
        yield "retry: 3000\n\n"
        while True:
            try:
                event = await asyncio.to_thread(q.get, True, 15.0)
            except queue.Empty:
                yield ": keep-alive\n\n"
                continue
            yield f"event: {event['type']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"
