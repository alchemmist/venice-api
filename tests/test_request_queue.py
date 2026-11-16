import asyncio
import unittest
from unittest.mock import patch

import main as bridge


class RequestQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_overlapping_calls_are_serialized(self):
        session = bridge.BrowserSession()
        entered = asyncio.Event()
        release = asyncio.Event()
        order = []

        async def first():
            async with session.request_slot():
                order.append("first")
                entered.set()
                await release.wait()

        async def second():
            async with session.request_slot():
                order.append("second")

        a = asyncio.create_task(first())
        await entered.wait()
        b = asyncio.create_task(second())
        await asyncio.sleep(0.01)
        self.assertEqual(order, ["first"])
        self.assertEqual(session._pending, 2)
        release.set()
        await asyncio.gather(a, b)
        self.assertEqual(order, ["first", "second"])
        self.assertEqual(session._pending, 0)

    async def test_capacity_is_bounded(self):
        session = bridge.BrowserSession()
        with patch.object(bridge, "MAX_PENDING_REQUESTS", 1):
            async with session.request_slot():
                with self.assertRaisesRegex(bridge.BrowserBusyError, "full"):
                    async with session.request_slot():
                        self.fail("Second request entered")
            self.assertEqual(session._pending, 0)

    async def test_wait_timeout_does_not_leak_capacity(self):
        session = bridge.BrowserSession()
        await session._lock.acquire()
        with patch.object(bridge, "QUEUE_TIMEOUT", 0.01):
            with self.assertRaisesRegex(bridge.BrowserBusyError, "timed out"):
                async with session.request_slot():
                    self.fail("Locked request entered")
        self.assertEqual(session._pending, 0)
        session._lock.release()
        async with session.request_slot():
            pass

    async def test_cancelled_waiter_releases_only_its_queue_slot(self):
        session = bridge.BrowserSession()
        await session._lock.acquire()
        async def waiting():
            async with session.request_slot():
                self.fail("Cancelled waiter entered")
        task = asyncio.create_task(waiting())
        await asyncio.sleep(0.01)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(session._pending, 0)
        self.assertTrue(session._lock.locked())
        session._lock.release()

    async def test_cancelled_active_request_releases_lock(self):
        session = bridge.BrowserSession()
        entered = asyncio.Event()
        async def active():
            async with session.request_slot():
                entered.set()
                await asyncio.Event().wait()
        task = asyncio.create_task(active())
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(session._pending, 0)
        self.assertFalse(session._lock.locked())


if __name__ == "__main__":
    unittest.main()
