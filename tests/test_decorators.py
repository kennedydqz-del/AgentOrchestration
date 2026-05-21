"""Tests for SDK decorators."""

import asyncio
import pytest
from src.sdk.decorators import task, agent, on_event


class TestTaskDecorator:
    @pytest.mark.asyncio
    async def test_async_task(self):
        @task(name="async-task")
        async def my_task() -> str:
            return "done"

        result = await my_task()
        assert result == "done"
        assert my_task.__task_config__["name"] == "async-task"

    @pytest.mark.asyncio
    async def test_sync_task(self):
        @task(name="sync-task")
        def my_task() -> str:
            return "done"

        result = await my_task()
        assert result == "done"
        assert my_task.__task_config__["name"] == "sync-task"

    @pytest.mark.asyncio
    async def test_sync_task_with_args(self):
        @task(name="sync-args")
        def my_task(greeting: str, name: str) -> str:
            return f"{greeting}, {name}!"

        result = await my_task("Hello", "World")
        assert result == "Hello, World!"

    @pytest.mark.asyncio
    async def test_async_task_timeout(self):
        @task(name="timeout-test", timeout=1)
        async def my_task():
            await asyncio.sleep(10)

        with pytest.raises(TimeoutError):
            await my_task()


class TestAgentDecorator:
    def test_agent_config(self):
        @agent(name="test-agent", version="2.0.0", description="A test agent")
        class MyAgent:
            pass

        assert MyAgent.__agent_config__["name"] == "test-agent"
        assert MyAgent.__agent_config__["version"] == "2.0.0"


class TestOnEventDecorator:
    @pytest.mark.asyncio
    async def test_async_event_handler(self):
        @on_event("test.event")
        async def handler(data: str) -> str:
            return f"processed: {data}"

        result = await handler("input")
        assert result == "processed: input"
        assert handler.__event_handler__ == "test.event"

    @pytest.mark.asyncio
    async def test_sync_event_handler(self):
        @on_event("sync.event")
        def handler(data: str) -> str:
            return f"sync: {data}"

        result = await handler("input")
        assert result == "sync: input"
        assert handler.__event_handler__ == "sync.event"
