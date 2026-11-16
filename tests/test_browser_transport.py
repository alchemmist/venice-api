import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import main as bridge


class BrowserTransportTests(unittest.IsolatedAsyncioTestCase):
    def route(self, path=None, prompt="fixture", method="POST"):
        route = MagicMock()
        route.continue_ = AsyncMock()
        route.abort = AsyncMock()
        route.request.method = method
        route.request.url = bridge.AGENT_INFERENCE_ORIGIN + (path or bridge.AGENT_ROUTE_SOURCE) + "?model=native"
        route.request.post_data_json = {
            "clientTools": [{"name": "approve_paid_usage"}],
            "input": {"messages": [{"id": "fixture-id", "type": "message", "status": "completed", "role": "user", "content": [{"type": "output_text", "text": prompt}]}]},
        }
        return route

    async def test_only_text_model_and_empty_tools_reach_backend(self):
        transport = bridge.AgentBrowserTransport("fixture")
        route = self.route()
        await transport.route(route)
        transport.check(complete=True)
        sent = route.continue_.call_args.kwargs
        self.assertEqual(sent["url"], bridge.AGENT_INFERENCE_ORIGIN + bridge.AGENT_ROUTE_TARGET)
        payload = json.loads(sent["post_data"])
        self.assertEqual(payload["tools"], [])
        self.assertEqual(payload["clientTools"], [])
        self.assertEqual(payload["modelId"], bridge.AGENT_TEXT_MODEL_ID)
        self.assertEqual(set(payload), {"tools", "clientTools", "modelId", "messages", "systemPrompt"})
        self.assertEqual(payload["systemPrompt"]["instructions"], bridge.AGENT_TRANSPORT_INSTRUCTIONS)
        self.assertEqual(payload["messages"][0]["content"][0]["text"], "fixture")
        route.abort.assert_not_awaited()

    async def test_plain_and_compaction_use_text_only_system_not_json_requirement(self):
        transport = bridge.AgentBrowserTransport("fixture", structured=False)
        route = self.route()
        await transport.route(route)
        payload = json.loads(route.continue_.call_args.kwargs["post_data"])
        self.assertEqual(payload["systemPrompt"]["instructions"], bridge.TEXT_TRANSPORT_INSTRUCTIONS)
        self.assertEqual(payload["tools"], [])

    async def test_browser_history_is_not_replayed(self):
        transport = bridge.AgentBrowserTransport("fixture")
        route = self.route()
        route.request.post_data_json["input"]["messages"].insert(0, {"role": "assistant", "content": "stale browser state"})
        await transport.route(route)
        payload = json.loads(route.continue_.call_args.kwargs["post_data"])
        self.assertEqual(len(payload["messages"]), 1)
        self.assertNotIn("stale browser state", json.dumps(payload))

    async def test_unknown_or_continuation_endpoints_are_blocked(self):
        for path in ["/api/inference/image", bridge.AGENT_ROUTE_TARGET, bridge.AGENT_ROUTE_SOURCE + "/changed"]:
            transport = bridge.AgentBrowserTransport("fixture")
            route = self.route(path)
            await transport.route(route)
            route.continue_.assert_not_awaited()
            route.abort.assert_awaited_once()
            with self.assertRaises(bridge.AgentTransportError):
                transport.check()

    async def test_repeated_native_tool_continuation_is_blocked(self):
        transport = bridge.AgentBrowserTransport("fixture")
        await transport.route(self.route())
        second = self.route()
        await transport.route(second)
        second.continue_.assert_not_awaited()
        second.abort.assert_awaited_once()
        with self.assertRaises(bridge.AgentTransportError):
            transport.check(complete=True)

    async def test_unexpected_composer_and_schema_changes_fail_closed(self):
        bodies = [{}, {"input": {"messages": []}}, {"input": {"messages": [{"role": "assistant", "content": []}]}},
                  {"input": {"messages": [{"role": "user", "content": [{"type": "output_text", "text": "different"}]}]}}]
        for body in bodies:
            transport = bridge.AgentBrowserTransport("fixture")
            route = self.route()
            route.request.post_data_json = body
            await transport.route(route)
            route.continue_.assert_not_awaited()
            route.abort.assert_awaited_once()
            with self.assertRaises(bridge.AgentTransportError):
                transport.check()

    async def test_foreign_origin_cannot_receive_host_transcript(self):
        transport = bridge.AgentBrowserTransport("fixture")
        route = self.route()
        route.request.url = "https://foreign.example" + bridge.AGENT_ROUTE_SOURCE
        await transport.route(route)
        route.continue_.assert_not_awaited()
        route.abort.assert_awaited_once()

    async def test_get_configuration_requests_are_unchanged(self):
        transport = bridge.AgentBrowserTransport("fixture")
        route = self.route("/api/inference/rate-limits", method="GET")
        await transport.route(route)
        route.continue_.assert_awaited_once_with()
        self.assertEqual(transport.rewritten, 0)

    async def test_missing_interception_cannot_be_reported_successful(self):
        with self.assertRaises(bridge.AgentTransportError):
            bridge.AgentBrowserTransport("fixture").check(complete=True)

    async def test_cancellation_removes_route_and_releases_slot(self):
        page = MagicMock()
        page.route = AsyncMock()
        page.unroute = AsyncMock()
        session = bridge.BrowserSession()
        session._page_or_start = AsyncMock(return_value=page)
        session._prepare = AsyncMock()
        session._generate_on_page = AsyncMock(side_effect=asyncio.CancelledError())
        with self.assertRaises(asyncio.CancelledError):
            await session.complete("fixture", structured=True)
        page.unroute.assert_awaited_once()
        self.assertEqual(session._pending, 0)
        self.assertFalse(session._lock.locked())

    async def test_cancellation_while_installing_route_still_cleans_up(self):
        page = MagicMock()
        page.route = AsyncMock(side_effect=asyncio.CancelledError())
        page.unroute = AsyncMock()
        session = bridge.BrowserSession()
        session._page_or_start = AsyncMock(return_value=page)
        session._prepare = AsyncMock()
        with self.assertRaises(asyncio.CancelledError):
            await session.complete("fixture", structured=True)
        page.unroute.assert_awaited_once()
        self.assertEqual(session._pending, 0)

    async def test_nonstream_transport_failure_is_explicit(self):
        request = bridge.ChatCompletionRequest(messages=[{"role": "user", "content": "fixture"}])
        with patch.object(bridge.SESSION, "complete", AsyncMock(side_effect=bridge.AgentTransportError("transport blocked"))):
            with self.assertRaises(bridge.HTTPException) as caught:
                await bridge.chat_completions(request)
        self.assertEqual(caught.exception.status_code, 502)
        self.assertEqual(caught.exception.detail, "transport blocked")

    async def test_unused_transport_cannot_emit_delta(self):
        page = MagicMock()
        page.route = AsyncMock()
        page.unroute = AsyncMock()
        session = bridge.BrowserSession()
        session._page_or_start = AsyncMock(return_value=page)
        session._prepare = AsyncMock()
        session._generate_on_page = AsyncMock(return_value='{"pi_bridge":1,"action":"answer","text":"unverified"}')
        delta = AsyncMock()
        with self.assertRaises(bridge.AgentTransportError):
            await session.complete("fixture", delta, structured=True)
        delta.assert_not_awaited()
        page.unroute.assert_awaited_once()

    async def test_stream_reports_transport_error_not_fake_final(self):
        request = bridge.ChatCompletionRequest(messages=[{"role": "user", "content": "fixture"}], stream=True)
        with patch.object(bridge.SESSION, "complete", AsyncMock(side_effect=bridge.AgentTransportError("transport blocked"))):
            response = await bridge.chat_completions(request)
            output = "".join([chunk async for chunk in response.body_iterator])
        self.assertIn("upstream_agent_transport_error", output)
        self.assertNotIn('"finish_reason": "stop"', output)
        self.assertNotIn('"tool_calls"', output)
