from collections import defaultdict
import asyncio
import logging


class EventEmitter:
    def __init__(self):
        self._events = defaultdict(list)
        # Handlers still running, so drain() can wait for them.
        self._handling: set[asyncio.Task] = set()

    def on(self, event_name, func=None):
        def decorator(f):
            self._events[event_name].append(f)
            return f

        if func:
            return decorator(func)
        return decorator

    async def _emit_async(self, event_name, *args, **kwargs):
        for callback in self._events[event_name]:
            await callback(*args, **kwargs)

    def emit(self, event_name, *args, **kwargs):
        task = asyncio.create_task(self._emit_async(event_name, *args, **kwargs))
        self._handling.add(task)
        task.add_done_callback(self._handler_done)

    def _handler_done(self, task):
        self._handling.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logging.error("Event handler failed", exc_info=task.exception())

    async def drain(self, timeout: float | None = None):
        """Wait until no handler is running, or `timeout` seconds pass.
        Handlers still running at the timeout keep running."""
        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout
        while self._handling:
            remaining = None if deadline is None else deadline - loop.time()
            if remaining is not None and remaining <= 0:
                return
            await asyncio.wait(set(self._handling), timeout=remaining)
