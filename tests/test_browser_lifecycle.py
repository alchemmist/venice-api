import asyncio
import gc
import json
import os
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch
from playwright.async_api import Browser, BrowserContext
import main as bridge

class BrowserLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._handlers = {}
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

    def _remember(self, obj, event, callback):
        self._handlers.setdefault(id(obj), {}).setdefault(event, []).append(callback)

    def _spec_context(self):
        ctx = create_autospec(BrowserContext, instance=True, spec_set=True)

        def on(event, callback):
            self._remember(ctx, event, callback)

        ctx.on.side_effect = on
        return ctx

    def _spec_browser(self, *contexts):
        browser = create_autospec(Browser, instance=True, spec_set=True)
        browser.is_connected.return_value = True
        browser.contexts = list(contexts)

        def on(event, callback):
            self._remember(browser, event, callback)

        browser.on.side_effect = on
        browser.new_context = AsyncMock(side_effect=AssertionError("CDP must not create a context"))
        return browser

    async def test_cdp_connect_skips_context_closed_event(self):
        session = bridge.BrowserSession()
        closed = self._spec_context()
        live = self._spec_context()

        def close_immediately(event, callback):
            if event == "close":
                callback()

        closed.on.side_effect = close_immediately
        browser = self._spec_browser(closed, live)
        pw = MagicMock()
        pw.chromium.connect_over_cdp = AsyncMock(return_value=browser)
        starter = MagicMock()
        starter.start = AsyncMock(return_value=pw)
        with patch.dict(os.environ, {"CDP_URL": "http://127.0.0.1:9"}), patch.object(bridge, "async_playwright", return_value=starter):
            await session._launch()
        self.assertIs(session._browser, browser)
        self.assertIs(session._ctx, live)
        self.assertFalse(session._context_is_open(closed))
        self.assertTrue(session._context_is_open(live))
        with self.assertRaises(AttributeError):
            closed.is_closed
        browser.new_context.assert_not_awaited()

    async def test_cdp_connect_fails_closed_without_persistent_context(self):
        session = bridge.BrowserSession()
        browser = self._spec_browser()
        pw = MagicMock()
        pw.stop = AsyncMock()
        pw.chromium.connect_over_cdp = AsyncMock(return_value=browser)
        starter = MagicMock()
        starter.start = AsyncMock(return_value=pw)
        with patch.dict(os.environ, {"CDP_URL": "http://127.0.0.1:9"}), patch.object(bridge, "async_playwright", return_value=starter):
            with self.assertRaisesRegex(RuntimeError, "no persistent context"):
                await session._launch()
        self.assertIsNone(session._ctx)
        self.assertIsNone(session._browser)
        pw.stop.assert_awaited()
        browser.new_context.assert_not_awaited()

    async def test_context_close_is_identity_bound(self):
        session = bridge.BrowserSession()
        browser = self._spec_browser()
        first = self._spec_context()
        second = self._spec_context()
        session._bind_browser(browser)
        session._bind_context(first)
        session._bind_context(second)
        self._handlers[id(first)]["close"][0]()
        self.assertFalse(session._context_is_open(first))
        self.assertTrue(session._context_is_open(second))
        self._handlers[id(browser)]["disconnected"][0]()
        self.assertFalse(session._context_is_open(second))

    async def test_browser_is_connected_false_rejects_context_without_is_closed(self):
        session = bridge.BrowserSession()
        browser = self._spec_browser()
        ctx = self._spec_context()
        session._bind_browser(browser)
        session._bind_context(ctx)
        browser.is_connected.return_value = False
        self.assertFalse(session._context_is_open(ctx))
        with self.assertRaises(AttributeError):
            ctx.is_closed

    async def test_cached_closed_cdp_context_is_not_reused(self):
        session = bridge.BrowserSession()
        browser = self._spec_browser()
        stale = self._spec_context()
        stale.pages = []
        stale.new_page = AsyncMock(side_effect=AssertionError("reused stale context"))
        stale.close = AsyncMock()
        session._bind_browser(browser)
        session._bind_context(stale)
        session._ctx = stale
        session._cdp = True
        self._handlers[id(stale)]["close"][0]()
        driver = AsyncMock()
        session._pw = driver
        fresh_page = MagicMock()
        fresh_page.is_closed.return_value = False
        fresh_page.url = bridge.AGENT_URL
        fresh = MagicMock()
        fresh.pages = []
        fresh.new_page = AsyncMock(return_value=fresh_page)

        async def launch():
            session._browser = None
            session._ctx = fresh
            session._cdp = True

        session._launch = AsyncMock(side_effect=launch)
        page = await session._page_or_start()
        self.assertIs(page, fresh_page)
        session._launch.assert_awaited_once()
        stale.new_page.assert_not_awaited()
        stale.close.assert_not_awaited()
        driver.stop.assert_awaited_once()

    async def test_recycle_opens_replacement_before_closing_last_page(self):
        session = bridge.BrowserSession()
        state = {"open": 1, "dead": False}
        new = MagicMock()
        context = MagicMock()

        async def new_page():
            if state["dead"]:
                raise RuntimeError("context died with last window")
            state["open"] += 1
            return new

        async def close():
            state["open"] -= 1
            if state["open"] <= 0:
                state["dead"] = True

        old = MagicMock(url=bridge.AGENT_URL)
        old.is_closed.return_value = False
        old.close = AsyncMock(side_effect=close)
        context.new_page = AsyncMock(side_effect=new_page)
        context.close = AsyncMock()
        session._page = old
        session._ctx = context
        session._authenticated = True
        self.assertIs(await session._page_or_start(recycle=True), new)
        self.assertFalse(state["dead"])
        self.assertEqual(state["open"], 1)
        self.assertIs(session._page, new)
        context.close.assert_not_awaited()

    async def test_transport_cleanup_does_not_mask_primary_error(self):
        session = bridge.BrowserSession()
        page = MagicMock()
        page.route = AsyncMock()
        page.unroute = AsyncMock(side_effect=RuntimeError("cleanup"))
        page.remove_listener = MagicMock()
        page.on = MagicMock()
        session._page_or_start = AsyncMock(return_value=page)
        session._prepare = AsyncMock()
        session._generate_on_page = AsyncMock(side_effect=RuntimeError("primary generation"))
        with self.assertRaisesRegex(RuntimeError, "primary generation"):
            await session.complete("prompt")
        session._generate_on_page.assert_awaited_once()

    async def test_transport_cleanup_failure_stays_fail_closed_without_replay(self):
        session = bridge.BrowserSession()
        page = MagicMock()
        page.route = AsyncMock()
        page.unroute = AsyncMock(side_effect=RuntimeError("cleanup"))
        page.remove_listener = MagicMock()
        page.on = MagicMock()
        session._page_or_start = AsyncMock(return_value=page)
        session._prepare = AsyncMock()
        session._generate_on_page = AsyncMock(return_value="ok")
        with patch.object(bridge.AgentBrowserTransport, "check", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "cleanup"):
                await session.complete("prompt")
        session._generate_on_page.assert_awaited_once()

    async def test_cookie_import_cleanup_does_not_replace_primary_error(self):
        session = bridge.BrowserSession()
        ctx = MagicMock()
        pw = MagicMock()
        pw.chromium.launch_persistent_context = AsyncMock(return_value=ctx)
        ctx.add_init_script = AsyncMock()
        starter = MagicMock()
        starter.start = AsyncMock(return_value=pw)
        session._apply_saved_cookies = AsyncMock(side_effect=RuntimeError("cookie primary"))
        session.close = AsyncMock(side_effect=RuntimeError("cleanup"))
        with patch.dict(os.environ, {"CDP_URL": ""}), patch.object(bridge, "async_playwright", return_value=starter), patch.object(bridge, "_chrome_profile", return_value=("/tmp/venice-lifecycle-test-profile", None)):
            with self.assertRaisesRegex(RuntimeError, "cookie primary"):
                await session._launch()
        session.close.assert_awaited_once()


    async def test_answer_bounds_hung_evaluate_without_placeholder(self):
        session = bridge.BrowserSession()
        messages = MagicMock()
        messages.count = AsyncMock(return_value=2)

        async def evaluate(expression, *args, **kwargs):
            await asyncio.sleep(1.0)
            return "[placeholder]"

        messages.last.evaluate = evaluate
        page = MagicMock()
        page.locator.return_value = messages
        started = time.monotonic()
        previous = bridge.EXTRACT_TIMEOUT
        bridge.EXTRACT_TIMEOUT = 0.05
        try:
            with self.assertRaises(TimeoutError):
                await session._answer(page, 1)
        finally:
            bridge.EXTRACT_TIMEOUT = previous
        self.assertLess(time.monotonic() - started, 0.4)

    async def test_nonstream_disconnect_cancels_browser_slot(self):
        session = bridge.BrowserSession()
        stop = MagicMock()
        stop.count = AsyncMock(return_value=1)
        stop.is_visible = AsyncMock(return_value=True)
        stop.click = AsyncMock()
        locator = MagicMock()
        locator.first = stop
        page = MagicMock()
        page.is_closed.return_value = False
        page.locator.return_value = locator
        session._page = page
        entered = asyncio.Event()
        calls = 0

        async def complete(prompt, on_delta=None, structured=False):
            nonlocal calls
            calls += 1
            async with session.request_slot():
                entered.set()
                await asyncio.Event().wait()
            return "should-not-complete"

        session.complete = complete
        body = json.dumps({
            "model": bridge.MODEL_ID,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        }).encode()
        state = {"body": False}

        async def receive():
            if not state["body"]:
                state["body"] = True
                return {"type": "http.request", "body": body, "more_body": False}
            await entered.wait()
            return {"type": "http.disconnect"}

        messages = []

        async def send(message):
            messages.append(message)

        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": "POST", "scheme": "http", "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions", "query_string": b"",
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
            "client": ("127.0.0.1", 9), "server": ("127.0.0.1", 80),
        }
        with patch.object(bridge, "SESSION", session), patch.object(bridge, "API_KEY", ""):
            await asyncio.wait_for(bridge.app(scope, receive, send), timeout=1.0)
        self.assertEqual(calls, 1)
        self.assertFalse(session._lock.locked())
        self.assertEqual(session._pending, 0)
        stop.click.assert_awaited()
        self.assertTrue(session._stop_acknowledged)
        statuses = [m.get("status") for m in messages if m.get("type") == "http.response.start"]
        self.assertNotIn(200, statuses)

    async def test_nonstream_external_cancel_during_wait_stops_generation(self):
        session = bridge.BrowserSession()
        page = MagicMock()
        page.is_closed.return_value = False
        page.locator.return_value.first.count = AsyncMock(return_value=0)
        session._page = page
        entered = asyncio.Event()
        calls = 0

        async def complete(prompt, on_delta=None, structured=False):
            nonlocal calls
            calls += 1
            async with session.request_slot():
                entered.set()
                await asyncio.Event().wait()
            return "should-not-complete"

        session.complete = complete
        body = json.dumps({
            "model": bridge.MODEL_ID,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        }).encode()
        state = {"body": False}

        async def receive():
            if not state["body"]:
                state["body"] = True
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.Event().wait()

        async def send(message):
            return None

        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": "POST", "scheme": "http", "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions", "query_string": b"",
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
            "client": ("127.0.0.1", 9), "server": ("127.0.0.1", 80),
        }
        with patch.object(bridge, "SESSION", session), patch.object(bridge, "API_KEY", ""):
            app_task = asyncio.create_task(bridge.app(scope, receive, send))
            await asyncio.wait_for(entered.wait(), timeout=1.0)
            app_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await app_task
        self.assertEqual(calls, 1)
        self.assertFalse(session._lock.locked())
        self.assertEqual(session._pending, 0)
        self.assertFalse(session._stop_acknowledged)

    def _composer_page(self):
        page = MagicMock()
        page.locator.return_value.count = AsyncMock(return_value=0)
        page.locator.return_value.first.click = AsyncMock()
        page.locator.return_value.first.fill = AsyncMock()
        page.keyboard.insert_text = AsyncMock()
        page.keyboard.press = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        return page

    def _clock(self):
        state = {"t": 1000.0}

        def monotonic():
            return state["t"]

        async def sleep(delay):
            state["t"] += delay

        return state, monotonic, sleep

    async def test_settle_past_deadline_is_refused(self):
        session = bridge.BrowserSession()
        page = self._composer_page()
        session._is_generating = AsyncMock(return_value=False)
        session._shot = AsyncMock()
        session._answer = AsyncMock(return_value="HI")
        transport = MagicMock(finished=True, request=object())
        transport.check = MagicMock()
        state, monotonic, sleep = self._clock()
        with patch.object(bridge, "GEN_TIMEOUT", 0.6), patch.object(bridge, "STABLE_SEC", 0.4), patch.object(bridge.time, "monotonic", monotonic), patch.object(bridge.asyncio, "sleep", sleep):
            with self.assertRaisesRegex(RuntimeError, "Generation deadline exceeded"):
                await session._generate_on_page(page, "prompt", False, transport)
        self.assertGreaterEqual(state["t"], 1000.6)

    async def test_settle_inside_deadline_returns_text(self):
        session = bridge.BrowserSession()
        page = self._composer_page()
        session._is_generating = AsyncMock(return_value=False)
        session._shot = AsyncMock()
        session._answer = AsyncMock(return_value="HI")
        transport = MagicMock(finished=True, request=object())
        transport.check = MagicMock()
        state, monotonic, sleep = self._clock()
        with patch.object(bridge, "GEN_TIMEOUT", 5), patch.object(bridge, "STABLE_SEC", 0.4), patch.object(bridge.time, "monotonic", monotonic), patch.object(bridge.asyncio, "sleep", sleep):
            self.assertEqual(await session._generate_on_page(page, "prompt", False, transport), "HI")
        self.assertLess(state["t"], 1005.0)

    async def test_missing_assistant_node_waits_like_pending_source(self):
        session = bridge.BrowserSession()
        page = self._composer_page()
        session._is_generating = AsyncMock(return_value=False)
        session._shot = AsyncMock()
        transport = MagicMock(finished=True, request=object())
        transport.check = MagicMock()
        state, monotonic, sleep = self._clock()

        async def answer(*_args):
            return "" if state["t"] < 1003.0 else "VALID"

        session._answer = answer
        with patch.object(bridge, "STABLE_SEC", 2), patch.object(bridge.time, "monotonic", monotonic), patch.object(bridge.asyncio, "sleep", sleep):
            self.assertEqual(await session._generate_on_page(page, "prompt", False, transport), "VALID")
        self.assertGreaterEqual(state["t"], 1003.0)

    async def test_identity_maps_do_not_retain_collected_browsers(self):
        session = bridge.BrowserSession()
        for _ in range(100):
            browser = MagicMock()
            browser.is_connected.return_value = True
            ctx = MagicMock()
            session._bind_browser(browser)
            session._bind_context(ctx)
        gc.collect()
        self.assertLessEqual(len(session._watched_ids), 2)
        self.assertLessEqual(len(session._context_browser_ids), 1)

    async def test_stop_count_hang_releases_slot_without_claiming_stop(self):
        session = bridge.BrowserSession()
        stop = MagicMock()

        async def count():
            await asyncio.Event().wait()

        stop.count = count
        stop.is_visible = AsyncMock(return_value=True)
        stop.click = AsyncMock()
        page = MagicMock()
        page.is_closed.return_value = False
        page.locator.return_value.first = stop
        session._page = page
        entered = asyncio.Event()

        async def active():
            async with session.request_slot():
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(active())
        await entered.wait()
        task.cancel()
        started = time.monotonic()
        with patch.object(bridge, "BROWSER_CLEANUP_TIMEOUT", 0.05):
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1.0)
        self.assertLess(time.monotonic() - started, 0.4)
        self.assertEqual(session._pending, 0)
        self.assertFalse(session._lock.locked())
        self.assertFalse(session._stop_acknowledged)
        stop.click.assert_not_awaited()

    async def test_unroute_hang_preserves_primary_cancellation(self):
        session = bridge.BrowserSession()
        page = MagicMock()
        page.route = AsyncMock()
        page.on = MagicMock()
        page.remove_listener = MagicMock()

        async def unroute(*_args, **_kwargs):
            await asyncio.Event().wait()

        page.unroute = unroute
        session._page_or_start = AsyncMock(return_value=page)
        session._prepare = AsyncMock()
        session._generate_on_page = AsyncMock(side_effect=asyncio.CancelledError())
        started = time.monotonic()
        with patch.object(bridge, "BROWSER_CLEANUP_TIMEOUT", 0.05):
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(session.complete("prompt"), 1.0)
        self.assertLess(time.monotonic() - started, 0.4)
        self.assertEqual(session._pending, 0)
        self.assertFalse(session._lock.locked())
        session._generate_on_page.assert_awaited_once()

    async def test_unroute_hang_preserves_primary_error(self):
        session = bridge.BrowserSession()
        page = MagicMock()
        page.route = AsyncMock()
        page.on = MagicMock()
        page.remove_listener = MagicMock()

        async def unroute(*_args, **_kwargs):
            await asyncio.Event().wait()

        page.unroute = unroute
        session._page_or_start = AsyncMock(return_value=page)
        session._prepare = AsyncMock()
        session._generate_on_page = AsyncMock(side_effect=RuntimeError("primary generation"))
        started = time.monotonic()
        with patch.object(bridge, "BROWSER_CLEANUP_TIMEOUT", 0.05):
            with self.assertRaisesRegex(RuntimeError, "primary generation"):
                await asyncio.wait_for(session.complete("prompt"), 1.0)
        self.assertLess(time.monotonic() - started, 0.4)
        self.assertEqual(session._pending, 0)
        self.assertFalse(session._lock.locked())
        session._generate_on_page.assert_awaited_once()

    async def test_new_page_hang_stays_inside_page_bound(self):
        session = bridge.BrowserSession()
        old = MagicMock(url=bridge.AGENT_URL)
        old.is_closed.return_value = False
        old.close = AsyncMock()
        context = MagicMock()

        async def new_page():
            await asyncio.Event().wait()

        context.new_page = new_page
        context.close = AsyncMock()
        session._page = old
        session._ctx = context
        session._authenticated = True
        started = time.monotonic()
        with patch.object(bridge, "BROWSER_PAGE_TIMEOUT", 0.05):
            with self.assertRaises(TimeoutError):
                await asyncio.wait_for(session._page_or_start(recycle=True), 1.0)
        self.assertLess(time.monotonic() - started, 0.4)
        self.assertIs(session._page, old)
        old.close.assert_not_awaited()
        context.close.assert_not_awaited()

    async def test_cdp_connect_failure_stops_local_driver(self):
        session = bridge.BrowserSession()
        pw = MagicMock()
        pw.stop = AsyncMock()
        pw.chromium.connect_over_cdp = AsyncMock(side_effect=RuntimeError("cdp refused"))
        starter = MagicMock()
        starter.start = AsyncMock(return_value=pw)
        with patch.dict(os.environ, {"CDP_URL": "http://127.0.0.1:9"}), patch.object(bridge, "async_playwright", return_value=starter):
            with self.assertRaisesRegex(RuntimeError, "cdp refused"):
                await session._launch()
        pw.stop.assert_awaited()
        self.assertIsNone(session._pw)
        self.assertIsNone(session._browser)
        self.assertIsNone(session._ctx)
        self.assertFalse(session._cdp)

    async def test_cdp_connect_hang_stops_local_driver(self):
        session = bridge.BrowserSession()
        pw = MagicMock()
        pw.stop = AsyncMock()

        async def connect(*_args, **_kwargs):
            await asyncio.Event().wait()

        pw.chromium.connect_over_cdp = connect
        starter = MagicMock()
        starter.start = AsyncMock(return_value=pw)
        started = time.monotonic()
        with patch.dict(os.environ, {"CDP_URL": "http://127.0.0.1:9"}), patch.object(bridge, "async_playwright", return_value=starter), patch.object(bridge, "BROWSER_PAGE_TIMEOUT", 0.05), patch.object(bridge, "BROWSER_CLEANUP_TIMEOUT", 0.05):
            with self.assertRaises(TimeoutError):
                await asyncio.wait_for(session._launch(), 1.0)
        self.assertLess(time.monotonic() - started, 0.4)
        pw.stop.assert_awaited()
        self.assertIsNone(session._pw)
        self.assertIsNone(session._browser)
        self.assertFalse(session._cdp)

    async def test_prepare_phase_hang_is_bounded(self):
        session = bridge.BrowserSession()
        page = MagicMock()
        session._page_or_start = AsyncMock(return_value=page)
        session._shot = AsyncMock()

        async def prepare(_page):
            await asyncio.Event().wait()

        session._prepare = prepare
        started = time.monotonic()
        with patch.object(bridge, "BROWSER_PREPARE_TIMEOUT", 0.05):
            with self.assertRaisesRegex(RuntimeError, "prepare phase exceeded"):
                await asyncio.wait_for(session.complete("prompt"), 1.0)
        self.assertLess(time.monotonic() - started, 0.4)
        self.assertEqual(session._pending, 0)
        self.assertFalse(session._lock.locked())
        session._shot.assert_awaited()

    async def test_generation_phase_hang_uses_start_plus_gen_budget(self):
        session = bridge.BrowserSession()
        page = self._composer_page()
        session._shot = AsyncMock()

        async def answer(*_args, **_kwargs):
            await asyncio.sleep(5)

        session._answer = answer
        transport = MagicMock(finished=False, request=None)
        transport.check = MagicMock()
        started = time.monotonic()
        with patch.object(bridge, "START_TIMEOUT", 0.05), patch.object(bridge, "GEN_TIMEOUT", 0.05):
            with self.assertRaisesRegex(RuntimeError, "Generation deadline exceeded"):
                await asyncio.wait_for(session._generate_on_page(page, "prompt", False, transport), 1.0)
        self.assertLess(time.monotonic() - started, 0.4)
        self.assertEqual(bridge.GEN_TIMEOUT, 600)
