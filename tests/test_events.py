from __future__ import annotations

import asyncio
import time

from aiolocal.events import EventBus


def test_event_bus_stream_does_not_block_the_loop():
    """Regression: a blocking queue.get inside the async generator used to freeze
    the whole server while any browser held the SSE connection open."""
    bus = EventBus()
    q = bus.subscribe()

    async def run():
        gen = bus.stream(q)
        assert await gen.__anext__() == "retry: 3000\n\n"

        pending = asyncio.create_task(gen.__anext__())
        started = time.perf_counter()
        await asyncio.sleep(0.2)
        elapsed = time.perf_counter() - started
        assert elapsed < 1.5, "event bus blocked the event loop"

        bus.publish("hello", {"value": 1})
        frame = await asyncio.wait_for(pending, timeout=2.0)
        assert "event: hello" in frame
        assert '"value": 1' in frame
        await gen.aclose()

    asyncio.run(run())
    bus.unsubscribe(q)


def test_event_bus_bounded_queue():
    bus = EventBus(max_queue=2)
    q = bus.subscribe()
    for i in range(6):
        bus.publish("tick", {"i": i})
    assert q.qsize() == 2
    first = q.get_nowait()
    second = q.get_nowait()
    assert (first["data"]["i"], second["data"]["i"]) == (4, 5)
    bus.unsubscribe(q)
