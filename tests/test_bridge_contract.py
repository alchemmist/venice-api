import asyncio
import re
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import main as bridge


class BridgeContractTests(unittest.IsolatedAsyncioTestCase):
    def test_model_match_is_exact(self):
        self.assertRegex("DeepSeek V4.1 Flash Uncensored", re.compile(bridge.MODEL_MATCH, re.I))
        for other in ["DeepSeek V4.1 Flash", "DeepSeek V4.1 Flash (1017)", "GLM Flash Uncensored"]:
            self.assertIsNone(re.search(bridge.MODEL_MATCH, other, re.I))

    async def test_unknown_model_is_rejected_before_browser(self):
        req = bridge.ChatCompletionRequest(model="not-the-real-model", messages=[{"role":"user","content":"hello"}])
        with self.assertRaises(bridge.HTTPException) as caught:
            await bridge.chat_completions(req)
        self.assertEqual(caught.exception.status_code, 400)

    async def test_missing_model_selector_fails_closed(self):
        session = bridge.BrowserSession()
        session._model_button = AsyncMock(return_value=None)
        with patch.object(bridge, "MODEL_SELECT_TIMEOUT", 0):
            with self.assertRaisesRegex(RuntimeError, "unknown model"):
                await session._ensure_model(MagicMock())

    async def test_model_selector_can_arrive_after_composer(self):
        session = bridge.BrowserSession()
        button = MagicMock()
        session._model_button = AsyncMock(side_effect=[None, None, button])
        with patch.object(bridge.asyncio, "sleep", AsyncMock()):
            self.assertIs(await session._wait_model_button(MagicMock()), button)
        self.assertEqual(session._model_button.await_count, 3)

    async def test_answer_does_not_strip_real_content(self):
        session = bridge.BrowserSession()
        messages = MagicMock()
        messages.count = AsyncMock(return_value=1)
        messages.last.evaluate = AsyncMock(return_value="assistant\nprivate\nDeepSeek is a model\nsources")
        page = MagicMock()
        page.locator.return_value = messages
        self.assertEqual(await session._answer(page, 0), "assistant\nprivate\nDeepSeek is a model\nsources")
        page.locator.assert_called_with(bridge.SEL_ASSISTANT)

    async def test_stream_buffers_dom_rewrites(self):
        session = bridge.BrowserSession()
        page = MagicMock()
        page.locator.return_value.count = AsyncMock(return_value=0)
        page.locator.return_value.first.click = AsyncMock()
        page.locator.return_value.first.fill = AsyncMock()
        page.keyboard.insert_text = AsyncMock()
        page.keyboard.press = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        page.route = AsyncMock()
        page.unroute = AsyncMock()
        session._page_or_start = AsyncMock(return_value=page)
        session._prepare = AsyncMock()
        session._answer = AsyncMock(side_effect=["partial wrong", "correct final", "correct final"])
        session._is_generating = AsyncMock(return_value=False)
        session._shot = AsyncMock()
        on_delta = AsyncMock()
        with patch.object(bridge, "STABLE_SEC", -1), patch.object(bridge.asyncio, "sleep", AsyncMock()), patch.object(bridge, "AgentBrowserTransport"):
            self.assertEqual(await session.complete("test", on_delta), "correct final")
        on_delta.assert_awaited_once_with("correct final")

    async def test_existing_messages_not_returned(self):
        page = MagicMock()
        page.locator.return_value.count = AsyncMock(return_value=1)
        self.assertEqual(await bridge.BrowserSession()._answer(page, 1), "")

    async def test_authenticated_session_does_not_login_again(self):
        page = MagicMock()
        page.wait_for_selector = AsyncMock()
        page.locator.return_value.count = AsyncMock(return_value=1)
        page.locator.return_value.inner_text = AsyncMock(return_value="Signed-in account")
        page.goto = AsyncMock()
        await bridge.BrowserSession()._ensure_login(page)
        page.goto.assert_not_called()

    async def test_guest_account_menu_is_not_authentication(self):
        page = MagicMock()
        page.wait_for_selector = AsyncMock()
        page.locator.return_value.count = AsyncMock(return_value=1)
        page.locator.return_value.inner_text = AsyncMock(return_value="Venice Guest")
        with patch.object(bridge, "AUTO_GOOGLE_LOGIN", False):
            with self.assertRaisesRegex(RuntimeError, "login required"):
                await bridge.BrowserSession()._ensure_login(page)


if __name__ == "__main__":
    unittest.main()
