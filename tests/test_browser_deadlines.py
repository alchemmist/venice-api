import asyncio
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

from playwright.async_api import BrowserContext, BrowserType, Page, Playwright
import main


class BrowserDeadlineTests(unittest.IsolatedAsyncioTestCase):
    async def hanging(self, *args, **kwargs):
        await asyncio.Event().wait()

    async def finishes(self, coroutine, exception):
        task = asyncio.create_task(coroutine)
        try:
            await asyncio.wait({task}, timeout=0.12)
            self.assertTrue(task.done(), 'browser RPC exceeded the configured bound')
            result = await asyncio.gather(task, return_exceptions=True)
            self.assertIsInstance(result[0], exception)
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_hung_stop_does_not_hold_cancelled_slot(self):
        session = main.BrowserSession()
        page = create_autospec(Page, instance=True)
        page.is_closed.return_value = False
        page.locator.return_value.first.count = AsyncMock(side_effect=self.hanging)
        session._page = page

        async def cancelled():
            async with session.request_slot():
                raise asyncio.CancelledError()

        with patch.object(main, 'BROWSER_CLEANUP_TIMEOUT', 0.02, create=True):
            await self.finishes(cancelled(), asyncio.CancelledError)
        self.assertFalse(session._lock.locked())
        self.assertEqual(session._pending, 0)

    async def test_hung_unroute_preserves_primary_and_releases_slot(self):
        session = main.BrowserSession()
        page = create_autospec(Page, instance=True)
        page.unroute.side_effect = self.hanging
        session._page_or_start = AsyncMock(return_value=page)
        session._prepare = AsyncMock()
        session._generate_on_page = AsyncMock(side_effect=ValueError('primary failure'))
        with patch.object(main, 'BROWSER_CLEANUP_TIMEOUT', 0.02, create=True):
            await self.finishes(session.complete('probe'), ValueError)
        self.assertFalse(session._lock.locked())
        self.assertEqual(session._pending, 0)

    async def test_hung_replacement_page_is_bounded(self):
        session = main.BrowserSession()
        page = create_autospec(Page, instance=True)
        page.url = main.AGENT_URL
        page.is_closed.return_value = False
        context = create_autospec(BrowserContext, instance=True)
        context.new_page.side_effect = self.hanging
        session._ctx = context
        session._page = page
        session._authenticated = True
        with patch.object(main, 'BROWSER_PAGE_TIMEOUT', 0.02, create=True):
            await self.finishes(session._page_or_start(recycle=True), TimeoutError)
        page.close.assert_not_awaited()

    async def test_cdp_connection_failure_releases_only_local_driver(self):
        session = main.BrowserSession()
        driver = create_autospec(Playwright, instance=True)
        driver.stop = AsyncMock()
        browser_type = create_autospec(BrowserType, instance=True)
        browser_type.connect_over_cdp.side_effect = ValueError('connection failed')
        driver.chromium = browser_type
        manager = MagicMock()
        manager.start = AsyncMock(return_value=driver)
        with patch.dict(os.environ, {'CDP_URL': 'http://127.0.0.1:9'}), patch.object(main, 'async_playwright', return_value=manager):
            with self.assertRaisesRegex(ValueError, 'connection failed'):
                await session._launch()
        driver.stop.assert_awaited_once()
        self.assertIsNone(session._pw)
        self.assertIsNone(session._ctx)

    async def test_hung_route_installation_is_bounded(self):
        session = main.BrowserSession()
        page = create_autospec(Page, instance=True)
        page.route.side_effect = self.hanging
        session._page_or_start = AsyncMock(return_value=page)
        session._prepare = AsyncMock()
        with patch.object(main, 'BROWSER_PAGE_TIMEOUT', 0.02):
            await self.finishes(session.complete('probe'), TimeoutError)
        self.assertFalse(session._lock.locked())
        self.assertEqual(session._pending, 0)

    async def test_cdp_connect_cancellation_releases_local_driver(self):
        session = main.BrowserSession()
        driver = create_autospec(Playwright, instance=True)
        driver.stop = AsyncMock()
        driver.chromium = create_autospec(BrowserType, instance=True)
        driver.chromium.connect_over_cdp.side_effect = self.hanging
        manager = MagicMock()
        manager.start = AsyncMock(return_value=driver)
        with patch.dict(os.environ, {'CDP_URL': 'http://127.0.0.1:9'}), patch.object(main, 'async_playwright', return_value=manager):
            task = asyncio.create_task(session._launch())
            await asyncio.sleep(0.02)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        driver.stop.assert_awaited_once()
        self.assertIsNone(session._pw)
