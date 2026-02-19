from collections import defaultdict
import asyncio

class EventEmitter:
    def __init__(self):
        self._events = defaultdict(list)

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
        asyncio.create_task(self._emit_async(event_name, *args, **kwargs))
