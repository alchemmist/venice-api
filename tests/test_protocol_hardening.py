import copy
import json
import math
import unittest
from unittest.mock import AsyncMock, patch

import agent_protocol
from agent_protocol import AgentProtocol, ProtocolError, PromptBudgetError, loads
import main


TOOLS = [{'type': 'function', 'function': {'name': 'read', 'parameters': {
    'type': 'object', 'properties': {'path': {'type': 'string'}},
    'required': ['path'], 'additionalProperties': False}}}]
USER = {'role': 'user', 'content': 'inspect fixture'}


def history(arguments='{"path":"fixture.txt"}', name='read'):
    return [copy.deepcopy(USER), {
        'role': 'assistant', 'content': None,
        'tool_calls': [{'id': 'call_fixture', 'type': 'function',
                        'function': {'name': name, 'arguments': arguments}}],
    }, {'role': 'tool', 'tool_call_id': 'call_fixture', 'content': 'ERROR: attempt failed'}]


class FiniteNumberTests(unittest.TestCase):
    def test_nested_invalid_numbers_and_duplicate_keys(self):
        for value in ['NaN', 'Infinity', '-Infinity', '1e999', '-1e999',
                      '{"duplicate":1,"duplicate":2}']:
            with self.subTest(value=value), self.assertRaises(ProtocolError):
                loads('{"a":[{"b":[{"c":' + value + '}]}]}')

    def test_deep_overflow_is_rejected(self):
        with self.assertRaises(ProtocolError):
            loads('[' * 70 + '1e999' + ']' * 70)

    def test_finite_extreme_values_remain_supported(self):
        result = loads('{"a":[1e308,-1e308,1e-999],"nested":{"value":3}}')
        self.assertTrue(all(math.isfinite(value) for value in result['a']))
        self.assertEqual(result['nested']['value'], 3)

    def test_native_nonfinite_input_rejected(self):
        for value in (float('nan'), float('inf'), float('-inf')):
            tools = copy.deepcopy(TOOLS)
            tools[0]['function']['parameters']['properties']['path']['default'] = value
            with self.subTest(value=value), self.assertRaises(ProtocolError):
                AgentProtocol(tools, 'auto', [USER])
            with self.assertRaises(ProtocolError):
                AgentProtocol(TOOLS, 'auto', [dict(USER, extra={'nested': [value]})])

    def test_history_numeric_overflow_rejected_even_with_error_result(self):
        with self.assertRaises(ProtocolError):
            AgentProtocol(TOOLS, 'auto', history('{"path":{"deep":[1e999]}}'))

    def test_new_call_numeric_overflow_never_emitted(self):
        protocol = AgentProtocol(TOOLS, 'auto', [USER])
        with self.assertRaises(ProtocolError):
            protocol.decode('{"pi_bridge":1,"action":"invoke","target":"read","payload":{"path":1e999}}')


class HistoryPolicyTests(unittest.TestCase):
    def test_schema_mismatch_and_error_replayed_without_coercion(self):
        messages = history('{"path":123}')
        original = copy.deepcopy(messages)
        protocol = AgentProtocol(TOOLS, 'auto', messages)
        self.assertEqual(protocol.messages, original)
        self.assertEqual(protocol.history_validation, [{
            'message_index': 1, 'tool_call_id': 'call_fixture', 'tool_name': 'read',
            'status': 'arguments_do_not_match_current_schema',
        }])
        self.assertIn('ERROR: attempt failed', protocol.prompt())
        self.assertEqual(json.loads(protocol.messages[1]['tool_calls'][0]['function']['arguments'])['path'], 123)
        with self.assertRaises(ProtocolError):
            protocol.decode('{"pi_bridge":1,"action":"invoke","target":"read","payload":{"path":123}}')

    def test_text_and_tool_calls_together_are_valid(self):
        messages = history()
        messages[1]['content'] = 'I will inspect the fixture.'
        protocol = AgentProtocol(TOOLS, 'auto', messages)
        self.assertEqual(protocol.messages[1]['content'], messages[1]['content'])
        self.assertEqual(protocol.history_validation, [])

    def test_old_tool_removed_or_schema_changed_is_not_execution_permission(self):
        messages = history(name='retired_read')
        protocol = AgentProtocol(TOOLS, 'auto', messages)
        self.assertEqual(protocol.history_validation[0]['status'], 'tool_not_declared_in_current_request')
        with self.assertRaises(ProtocolError):
            protocol.decode('{"pi_bridge":1,"action":"invoke","target":"retired_read","payload":{"path":"fixture.txt"}}')
        # Compaction may replay completed history with no tools currently enabled.
        AgentProtocol([], 'auto', messages).prompt()

    def test_malformed_history_shapes_are_rejected(self):
        variants = []
        for change in [
            lambda m: m[1].update(tool_calls={'wrong': 'shape'}),
            lambda m: m[1]['tool_calls'].__setitem__(0, 'not an object'),
            lambda m: m[1]['tool_calls'][0]['function'].update(name=''),
            lambda m: m[1]['tool_calls'][0]['function'].update(name=42),
            lambda m: m[1]['tool_calls'][0]['function'].update(arguments={}),
            lambda m: m[1]['tool_calls'][0]['function'].update(arguments='[]'),
            lambda m: m[1]['tool_calls'][0]['function'].update(extra='unsupported'),
            lambda m: m[1]['tool_calls'][0].update(extra='unsupported'),
            lambda m: m[0].update(tool_call_id='call_fixture'),
            lambda m: m[0].update(tool_calls=[]),
            lambda m: m[2].update(content=None),
            lambda m: m[2].update(tool_call_id='unmatched'),
            lambda m: m.insert(2, dict(USER)),
        ]:
            messages = history()
            change(messages)
            variants.append(messages)
        variants += [[{'role': 'assistant', 'content': None}], [USER, None]]
        for messages in variants:
            with self.subTest(messages=messages), self.assertRaises(ProtocolError):
                AgentProtocol(TOOLS, 'auto', messages)


class BudgetTests(unittest.TestCase):
    def test_budget_counts_utf8_and_does_not_disclose_or_trim_content(self):
        messages = history()
        messages[2]['content'] = 'PRIVATE-CANARY-' + '\u044f' * 60000
        original = copy.deepcopy(messages)
        protocol = AgentProtocol(TOOLS, 'auto', messages)
        with self.assertRaises(PromptBudgetError) as caught:
            protocol.prompt()
        error = caught.exception.as_openai_error()
        budget = error['budget']
        self.assertEqual(error['code'], 'context_length_exceeded')
        self.assertEqual(budget['largest_messages'][0]['index'], 2)
        self.assertEqual(budget['largest_messages'][0]['role'], 'tool')
        self.assertIn('#2 tool=', error['message'])
        expected = sum(len(json.dumps(m, ensure_ascii=False, allow_nan=False).encode()) for m in messages)
        self.assertEqual(budget['messages_bytes'], expected)
        self.assertEqual(budget['total_bytes'], budget['messages_bytes'] + budget['non_message_bytes'])
        self.assertNotIn('PRIVATE-CANARY', json.dumps(error))
        self.assertNotIn('call_fixture', json.dumps(error))
        self.assertEqual(protocol.messages, original)

    def test_exact_byte_boundary(self):
        protocol = AgentProtocol(TOOLS, 'auto', [USER])
        text = protocol.prompt()
        count = len(text.encode())
        with patch.object(agent_protocol, 'MAX_AGENT_PROMPT_BYTES', count):
            self.assertEqual(protocol.prompt(), text)
        with patch.object(agent_protocol, 'MAX_AGENT_PROMPT_BYTES', count - 1):
            with self.assertRaises(PromptBudgetError):
                protocol.prompt()

    def test_tool_definition_overhead_has_distinct_guidance(self):
        tools = copy.deepcopy(TOOLS)
        tools[0]['function']['description'] = 'schema-description-' * 10000
        with self.assertRaises(PromptBudgetError) as caught:
            AgentProtocol(tools, 'auto', [USER]).prompt()
        error = caught.exception.as_openai_error()
        self.assertEqual(error['param'], 'tools')
        self.assertIn('compaction alone cannot fit', error['message'])


class BudgetHttpTests(unittest.IsolatedAsyncioTestCase):
    async def test_budget_error_is_typed_before_browser_or_sse_headers(self):
        for stream in (False, True):
            messages = history()
            messages[2]['content'] = 'PRIVATE-CANARY-' + 'x' * 120000
            req = main.ChatCompletionRequest(messages=messages, tools=TOOLS, stream=stream)
            with patch.object(main.SESSION, 'complete', AsyncMock()) as complete:
                response = await main.chat_completions(req)
            self.assertEqual(response.status_code, 400)
            error = json.loads(response.body)['error']
            self.assertEqual(error['type'], 'invalid_request_error')
            self.assertEqual(error['code'], 'context_length_exceeded')
            self.assertIn('Compact history in Pi', error['message'])
            self.assertNotIn('PRIVATE-CANARY', response.body.decode())
            complete.assert_not_called()
