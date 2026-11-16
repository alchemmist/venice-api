import asyncio
import base64
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import main
from agent_protocol import AgentProtocol, PromptBudgetError

TOOLS = [{'type': 'function', 'function': {'name': 'read', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path'], 'additionalProperties': False}}}]
CALL = '{"pi_bridge":1,"action":"invoke","target":"read","payload":{"path":"fixture.txt"}}'
BAD = 'PRIVATE_REJECTED_REPLY: ```not an envelope```'

class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'records'
        self.env = patch.dict(os.environ, {'AGENT_REJECTION_DIR': str(self.root)})
        self.env.start()
    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()
    def req(self, stream=False):
        return main.ChatCompletionRequest(messages=[{'role':'user','content':'read fixture'}], tools=TOOLS, stream=stream)
    async def test_nonstream_regeneration_is_visible_and_usage_counts_both(self):
        with patch.object(main.SESSION, 'complete', AsyncMock(side_effect=[BAD, CALL])) as complete:
            response = await main.chat_completions(self.req())
        message = response['choices'][0]['message']
        self.assertEqual(len(message['tool_calls']), 1)
        self.assertIn('protocol regeneration 1/1', message['content'])
        self.assertNotIn('PRIVATE_REJECTED_REPLY', json.dumps(response))
        self.assertEqual(message['bridge_protocol_recovery']['attempts'], 1)
        self.assertEqual(complete.await_count, 2)
        a, b = [call.args[0] for call in complete.await_args_list]
        self.assertIn('NO Pi host tools were executed', b)
        self.assertNotIn(BAD, b)
        self.assertEqual(response['usage'], main._usage(a+b, BAD+CALL))
    async def test_stream_has_notice_then_one_valid_call_and_no_bad_prefix(self):
        replies = iter([BAD, CALL])
        async def complete(prompt, on_delta=None, **kwargs):
            text = next(replies)
            if on_delta: await on_delta(text)
            return text
        with patch.object(main.SESSION, 'complete', complete):
            response = await main.chat_completions(self.req(True))
            output = ''.join([x async for x in response.body_iterator])
        self.assertEqual(output.count('protocol regeneration 1/1'), 1)
        self.assertEqual(output.count('"tool_calls": ['), 1)
        self.assertNotIn('PRIVATE_REJECTED_REPLY', output)
        self.assertIn('diagnostic_id=', output)
        self.assertIn('[DONE]', output)
    async def test_second_invalid_reply_stops_and_both_are_recorded(self):
        with patch.object(main.SESSION, 'complete', AsyncMock(return_value=BAD)) as complete:
            with self.assertRaises(main.HTTPException) as error:
                await main.chat_completions(self.req())
        self.assertEqual(error.exception.status_code, 502)
        self.assertEqual(complete.await_count, 2)
        self.assertIn('diagnostic_id=', error.exception.detail)
        self.assertNotIn(BAD, error.exception.detail)
        records = list(self.root.glob('*.json'))
        self.assertEqual(len(records), 2)
        self.assertEqual(stat.S_IMODE(self.root.stat().st_mode), 0o700)
        for path in records:
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            record=json.loads(path.read_text())
            self.assertEqual(base64.b64decode(record['raw_utf8_base64']).decode(), BAD)
            self.assertNotIn('prompt', record)
    async def test_cancel_during_regeneration_propagates_without_third_attempt(self):
        with patch.object(main.SESSION, 'complete', AsyncMock(side_effect=[BAD, asyncio.CancelledError])) as complete:
            with self.assertRaises(asyncio.CancelledError):
                await main.chat_completions(self.req())
        self.assertEqual(complete.await_count, 2)
    async def test_transport_errors_are_not_regenerated(self):
        with patch.object(main.SESSION, 'complete', AsyncMock(side_effect=main.AgentTransportError('blocked'))) as complete:
            with self.assertRaises(main.HTTPException): await main.chat_completions(self.req())
        self.assertEqual(complete.await_count, 1)
        self.assertFalse(self.root.exists())
    async def test_no_regeneration_if_new_instruction_exceeds_budget(self):
        original = AgentProtocol.prompt
        def prompt(obj, *, recovery=False):
            if recovery: raise PromptBudgetError(120001, [], 1000)
            return original(obj)
        with patch.object(AgentProtocol, 'prompt', prompt), patch.object(main.SESSION, 'complete', AsyncMock(return_value=BAD)) as complete:
            with self.assertRaises(main.HTTPException): await main.chat_completions(self.req())
        self.assertEqual(complete.await_count, 1)
    async def test_valid_false_tool_failure_story_is_not_filtered_or_retried(self):
        story='Tool not found: fabricated story with NO actual host failure'
        raw=json.dumps({'pi_bridge':1,'action':'answer','text':story})
        with patch.object(main.SESSION, 'complete', AsyncMock(return_value=raw)) as complete:
            response=await main.chat_completions(self.req())
        self.assertEqual(response['choices'][0]['message']['content'],story)
        self.assertEqual(complete.await_count,1)
    async def test_source_failure_is_typed_but_literal_marker_is_valid_text(self):
        page=MagicMock()
        page.locator.return_value.count=AsyncMock(return_value=1)
        page.locator.return_value.last.evaluate=AsyncMock(return_value=None)
        session=main.BrowserSession()
        self.assertIsNone(await session._answer(page,0))
        page.locator.return_value.last.evaluate=AsyncMock(return_value='INVALID_AGENT_SOURCE')
        self.assertEqual(await session._answer(page,0),'INVALID_AGENT_SOURCE')
    async def test_pending_source_is_not_a_fatal_extraction_failure(self):
        page=MagicMock()
        page.locator.return_value.count=AsyncMock(return_value=0)
        page.locator.return_value.first.click=AsyncMock()
        page.locator.return_value.first.fill=AsyncMock()
        page.locator.return_value.click=AsyncMock()
        page.locator.return_value.fill=AsyncMock()
        page.locator.return_value.press=AsyncMock()
        page.keyboard.insert_text=AsyncMock()
        page.keyboard.press=AsyncMock()
        page.wait_for_timeout=AsyncMock()
        page.route=AsyncMock()
        page.unroute=AsyncMock()
        session=main.BrowserSession()
        session._page_or_start=AsyncMock(return_value=page)
        session._prepare=AsyncMock()
        session._is_generating=AsyncMock(return_value=False)
        session._shot=AsyncMock()
        transport=MagicMock(finished=False)
        transport.route=AsyncMock()
        count=0
        async def answer(*args):
            nonlocal count
            count+=1
            if count<3:return None
            transport.finished=True
            return CALL
        session._answer=answer
        with patch.object(main, 'AgentBrowserTransport', return_value=transport), patch.object(main, 'STABLE_SEC', -1), patch.object(main.asyncio, 'sleep', AsyncMock()):
            self.assertEqual(await session.complete('fixture',structured=True),CALL)
        self.assertGreaterEqual(count,4)
        session._answer=AsyncMock(return_value=None)
        with patch.object(main, 'AgentBrowserTransport', return_value=transport), patch.object(main, 'STABLE_SEC', -1), patch.object(main, 'max', return_value=-1, create=True), patch.object(main.asyncio, 'sleep', AsyncMock()):
            with self.assertRaises(main.AgentTransportError): await session.complete('fixture',structured=True)
        self.assertFalse(session._lock.locked())

    def test_diagnostics_are_bounded_and_unsafe_directory_is_refused(self):
        for i in range(11): main._record_rejection('x'*70000,'invalid','fixture',i)
        paths=list(self.root.glob('*.json'))
        self.assertEqual(len(paths),8)
        record=json.loads(paths[-1].read_text())
        self.assertEqual(len(base64.b64decode(record['raw_utf8_base64'])),65536)
        self.assertTrue(record['truncated'])
        self.root.chmod(0o755)
        self.assertIsNone(main._record_rejection(BAD,'invalid','fixture',12))
        self.assertEqual(len(list(self.root.glob('*.json'))),8)
