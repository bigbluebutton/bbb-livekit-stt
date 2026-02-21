import asyncio

from events import EventEmitter


class TestEventEmitter:
    async def test_on_decorator_registers_callback(self):
        emitter = EventEmitter()
        results = []

        @emitter.on("test_event")
        async def handler(**kwargs):
            results.append(kwargs.get("value"))

        emitter.emit("test_event", value=42)
        await asyncio.sleep(0)

        assert results == [42]

    async def test_on_direct_function_registers_callback(self):
        emitter = EventEmitter()
        results = []

        async def handler(**kwargs):
            results.append(kwargs.get("value"))

        emitter.on("test_event", handler)
        emitter.emit("test_event", value="hello")
        await asyncio.sleep(0)

        assert results == ["hello"]

    async def test_emit_fires_all_registered_callbacks(self):
        emitter = EventEmitter()
        call_order = []

        @emitter.on("click")
        async def first(**kwargs):
            call_order.append("first")

        @emitter.on("click")
        async def second(**kwargs):
            call_order.append("second")

        emitter.emit("click")
        await asyncio.sleep(0)

        assert call_order == ["first", "second"]

    async def test_emit_unregistered_event_does_not_raise(self):
        emitter = EventEmitter()
        # Emitting an event with no listeners should be a no-op
        emitter.emit("no_listeners")
        await asyncio.sleep(0)

    async def test_emit_passes_kwargs_to_callbacks(self):
        emitter = EventEmitter()
        received = {}

        @emitter.on("data")
        async def handler(**kwargs):
            received.update(kwargs)

        emitter.emit("data", participant="alice", locale="en-US")
        await asyncio.sleep(0)

        assert received == {"participant": "alice", "locale": "en-US"}

    async def test_different_events_do_not_cross_fire(self):
        emitter = EventEmitter()
        a_calls = []
        b_calls = []

        @emitter.on("event_a")
        async def on_a(**kwargs):
            a_calls.append(True)

        @emitter.on("event_b")
        async def on_b(**kwargs):
            b_calls.append(True)

        emitter.emit("event_a")
        await asyncio.sleep(0)

        assert a_calls == [True]
        assert b_calls == []
