import json
import unittest
from unittest.mock import AsyncMock, patch

from agent_protocol import AgentProtocol, ProtocolError
import main

TOOLS = [{'type': 'function', 'function': {'name': 'read', 'parameters': {
    'type': 'object', 'properties': {'path': {'type': 'string'}},
    'required': ['path'], 'additionalProperties': False}}}]
CALL = '{"version":1,"type":"tool_call","name":"read","arguments":{"path":"fixture.txt"}}'
FINAL = '{"version":1,"type":"final","content":"done"}'


class ProtocolTests(unittest.TestCase):
    def protocol(self, choice='auto', messages=None):
        return AgentProtocol(TOOLS, choice, messages or [{'role': 'user', 'content': 'read fixture'}])

    def test_valid_call(self):
        message, finish = self.protocol().decode(CALL)
        self.assertEqual(finish, 'tool_calls')
        self.assertIsNone(message['content'])
        self.assertEqual(json.loads(message['tool_calls'][0]['function']['arguments']), {'path': 'fixture.txt'})

    def test_invalid_outputs_fail_closed(self):
        for text in ['```json\n'+CALL+'\n```', 'Here is '+CALL, CALL[:-1],
                     CALL.replace('"read"', '"bash"'), CALL.replace('"fixture.txt"', '123'),
                     CALL.replace('"path":', '"extra":1,"path":'),
                     CALL.replace('"version":1', '"version":1,"version":1'),
                     CALL.replace('"version":1', '"version":true'), 'x'*65537]:
            with self.subTest(text=text[:90]), self.assertRaises(ProtocolError):
                self.protocol().decode(text)

    def test_choice(self):
        with self.assertRaises(ProtocolError):
            self.protocol('none').decode(CALL)
        with self.assertRaises(ProtocolError):
            self.protocol('required').decode(FINAL)
        self.protocol({'type': 'function', 'function': {'name': 'read'}}).decode(CALL)
        self.assertEqual(self.protocol('none').decode(FINAL)[1], 'stop')

    def test_history_preserved(self):
        message, _ = self.protocol().decode(CALL)
        result = {'role': 'tool', 'tool_call_id': message['tool_calls'][0]['id'], 'content': 'ERROR: file absent'}
        p = self.protocol(messages=[message, result])
        self.assertIn('ERROR: file absent', p.prompt())
        self.assertIn(result['tool_call_id'], p.prompt())
        with self.assertRaises(ProtocolError):
            self.protocol(messages=[message])

    def test_unsupported_blocks(self):
        with self.assertRaises(ProtocolError):
            self.protocol(messages=[{'role': 'user', 'content': [{'type': 'image_url'}]}])

    def test_prompt_budget(self):
        with self.assertRaises(ProtocolError):
            self.protocol(messages=[{'role': 'user', 'content': 'x'*120001}]).prompt()


class ResponseTests(unittest.IsolatedAsyncioTestCase):
    async def test_nonstream_and_stream(self):
        for stream in (False, True):
            req = main.ChatCompletionRequest(messages=[{'role': 'user', 'content': 'read'}], tools=TOOLS, stream=stream)
            async def complete(prompt, on_delta=None):
                if on_delta:
                    await on_delta(CALL)
                return CALL
            with patch.object(main.SESSION, 'complete', complete):
                response = await main.chat_completions(req)
                if stream:
                    data = ''.join([c async for c in response.body_iterator])
                    self.assertNotIn('"content": "{', data)
                    events = [json.loads(x[6:]) for x in data.split('\n\n') if x.startswith('data: {')]
                    call = events[1]['choices'][0]['delta']['tool_calls'][0]
                    self.assertEqual(call['index'], 0)
                    self.assertEqual(call['function']['name'], 'read')
                    self.assertEqual(events[-1]['choices'][0]['finish_reason'], 'tool_calls')
                    self.assertTrue(data.endswith('data: [DONE]\n\n'))
                else:
                    self.assertEqual(response['choices'][0]['finish_reason'], 'tool_calls')

    async def test_plain_text_unchanged(self):
        with patch.object(main.SESSION, 'complete', AsyncMock(return_value='pong')):
            r = await main.chat_completions(main.ChatCompletionRequest(messages=[{'role':'user','content':'ping'}]))
            self.assertEqual(r['choices'][0]['message']['content'], 'pong')
