import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import main as bridge

class BrowserLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_recycle_releases_only_owned_page_and_retains_context(self):
        session = bridge.BrowserSession()
        old = MagicMock(url=bridge.AGENT_URL)
        old.is_closed.return_value = False
        old.close = AsyncMock()
        new = MagicMock()
        context = MagicMock()
        context.new_page = AsyncMock(return_value=new)
        context.close = AsyncMock()
        session._page = old
        session._ctx = context
        session._authenticated = True
        self.assertIs(await session._page_or_start(recycle=True), new)
        old.close.assert_awaited_once()
        context.new_page.assert_awaited_once()
        context.close.assert_not_awaited()
        self.assertTrue(session._authenticated)
        self.assertIs(session._ctx, context)

    async def test_unknown_authentication_does_not_discard_operator_page(self):
        session = bridge.BrowserSession()
        page = MagicMock(url=bridge.AGENT_URL)
        page.is_closed.return_value = False
        page.close = AsyncMock()
        session._page = page
        self.assertIs(await session._page_or_start(recycle=True), page)
        page.close.assert_not_awaited()

    async def test_busy_health_does_not_touch_navigating_page(self):
        session = bridge.BrowserSession()
        session._authenticated = True
        page = MagicMock()
        session._page = page
        async with session._lock:
            with patch.object(bridge, 'SESSION', session):
                response = await bridge.healthz()
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'"busy":true', response.body)
        page.locator.assert_not_called()

    async def test_probe_failure_does_not_close_shared_transport(self):
        session = bridge.BrowserSession()
        page = MagicMock()
        page.is_closed.return_value = False
        page.locator.return_value.inner_text = AsyncMock(side_effect=asyncio.TimeoutError)
        session._page = page
        session.close = AsyncMock()
        with patch.object(bridge, 'SESSION', session):
            response = await bridge.healthz()
        self.assertEqual(response.status_code, 503)
        session.close.assert_not_awaited()
        self.assertFalse(session._lock.locked())

    async def test_uninitialized_probe_does_not_launch_a_browser(self):
        session = bridge.BrowserSession()
        session._page_or_start = AsyncMock()
        with patch.object(bridge, 'SESSION', session):
            response = await bridge.healthz()
        self.assertEqual(response.status_code, 503)
        session._page_or_start.assert_not_awaited()
