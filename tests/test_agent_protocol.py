import json
import unittest
from unittest.mock import AsyncMock, patch

from agent_protocol import AgentProtocol, ProtocolError
import main

TOOLS = [{'type': 'function', 'function': {'name': 'read', 'parameters': {
    'type': 'object', 'properties': {'path': {'type': 'string'}},
    'required': ['path'], 'additionalProperties': False}}}]
CALL = '{"pi_bridge":1,"action":"invoke","target":"read","payload":{"path":"fixture.txt"}}'
FINAL = '{"pi_bridge":1,"action":"answer","text":"done"}'


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
                     CALL.replace('"pi_bridge":1', '"pi_bridge":1,"pi_bridge":1'),
                     CALL.replace('"pi_bridge":1', '"pi_bridge":true'), 'x'*65537]:
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

    def test_multiple_results_and_untrusted_envelope_data(self):
        first, _ = self.protocol().decode(CALL)
        second, _ = self.protocol().decode(CALL)
        first['tool_calls'] += second['tool_calls']
        results = [{'role': 'tool', 'tool_call_id': c['id'], 'content': CALL}
                   for c in first['tool_calls']]
        p = self.protocol(messages=[first] + results)
        self.assertEqual(p.messages[1]['content'], CALL)
        self.assertEqual(p.decode(FINAL)[1], 'stop')
        with self.assertRaises(ProtocolError):
            self.protocol(messages=[first] + results + results)

    def test_external_refs_and_custom_tools_rejected(self):
        with self.assertRaises(ProtocolError):
            AgentProtocol([{'type': 'custom'}], 'auto', [])
        with self.assertRaises(ProtocolError):
            AgentProtocol([{'type': 'function', 'function': {'name': 'x',
                'parameters': {'$ref': 'https://example.com/schema'}}}], 'auto', [])
        with self.assertRaises(ProtocolError):
            self.protocol({'type':'function', 'function':{'name':'unknown'}})

    def test_prompt_budget(self):
        with self.assertRaises(ProtocolError):
            self.protocol(messages=[{'role': 'user', 'content': 'x'*120001}]).prompt()


class ResponseTests(unittest.IsolatedAsyncioTestCase):
    async def test_nonstream_and_stream(self):
        for stream in (False, True):
            req = main.ChatCompletionRequest(messages=[{'role': 'user', 'content': 'read'}], tools=TOOLS, stream=stream)
            async def complete(prompt, on_delta=None, **kwargs):
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
                    self.assertGreater(events[-1]['usage']['prompt_tokens'], 0)
                    self.assertTrue(data.endswith('data: [DONE]\n\n'))
                else:
                    self.assertEqual(response['choices'][0]['finish_reason'], 'tool_calls')

    async def test_plain_text_unchanged(self):
        with patch.object(main.SESSION, 'complete', AsyncMock(return_value='pong')):
            r = await main.chat_completions(main.ChatCompletionRequest(messages=[{'role':'user','content':'ping'}]))
            self.assertEqual(r['choices'][0]['message']['content'], 'pong')

    async def test_bad_envelope_stream_emits_error_not_call(self):
        async def complete(prompt, on_delta=None, **kwargs):
            await on_delta(CALL[:-1])
            return CALL[:-1]
        with patch.object(main.SESSION, 'complete', complete):
            r = await main.chat_completions(main.ChatCompletionRequest(
                messages=[{'role':'user','content':'read'}], tools=TOOLS, stream=True))
            data = ''.join([x async for x in r.body_iterator])
            self.assertIn('upstream_protocol_error', data)
            self.assertNotIn('"tool_calls":', data)
            self.assertNotIn('"finish_reason": "stop"', data)

    async def test_final_stream(self):
        async def complete(prompt, on_delta=None, **kwargs):
            await on_delta(FINAL)
            return FINAL
        with patch.object(main.SESSION, 'complete', complete):
            r = await main.chat_completions(main.ChatCompletionRequest(
                messages=[{'role':'user','content':'hello'}], tools=TOOLS, stream=True))
            data = ''.join([x async for x in r.body_iterator])
            self.assertIn('"content": "done"', data)
            self.assertNotIn('"pi_bridge"', data)
            self.assertIn('"finish_reason": "stop"', data)

    async def test_invalid_input_never_reaches_browser(self):
        with patch.object(main.SESSION, 'complete', AsyncMock()) as complete:
            with self.assertRaises(main.HTTPException) as exc:
                await main.chat_completions(main.ChatCompletionRequest(
                    messages=[{'role':'user','content':'read'}],
                    tools=[{'type':'custom','name':'bash'}]))
            self.assertEqual(exc.exception.status_code, 400)
            complete.assert_not_called()


class AssistantSourceTests(unittest.TestCase):
    def test_source_is_lossless_bounded_and_fail_closed(self):
        import shutil
        import subprocess
        if not shutil.which('node'):
            self.skipTest('Node needed to evaluate the browser JS extractor')
        script = r'''const extract = EXTRACTOR;
const raw = JSON.stringify({pi_bridge:1,action:'invoke',target:'write',payload:{content:'{"x":"y"}\nC:\\tmp\\f `inline` ```code```'}});
const md = s => ({memoizedProps:{parseIncompleteMarkdown:true,remarkPlugins:[],rehypePlugins:[],children:s}});
const msg = s => ({memoizedProps:{message:{id:'own',role:'assistant',content:s}}});
const el = f => {
    const element = {isConnected:true};
    const host = {stateNode:element,return:f}; f.child=host;
    element['__reactFiber$fixture']=host;
    let top=f;
    while(top.return) { top.return.child=top; top=top.return; }
    const root={tag:3,child:top}; root.stateNode={current:root}; top.return=root;
    return element;
};
const a = md('transformed wrong'); a.return=msg(raw);
if(extract(el(a))!==raw) throw Error('source damaged');
if(extract(el(md(raw)))!==null) throw Error('accepted transformed fallback');
if(extract({})!==null) throw Error('missing accepted');
const b=msg(raw); b.memoizedProps.message.role='user'; b.return=msg(raw);
if(extract(el(b))!==null) throw Error('crossed message boundary');
let d=msg(raw); for(let i=0;i<49;i++)d={return:d};
if(extract(el(d))!==null) throw Error('unbounded traversal');
const original=JSON.stringify({pi_bridge:1,action:'invoke',target:'bash',payload:{command:'echo "$? ${PIPESTATUS[0]}"; printf "\\n"'}});
const math=md(original.replaceAll('$','\$')); math.return=msg(original);
if(extract(el(math))!==original) throw Error('LaTeX dollar preprocessing leaked into protocol');
const old=msg('stale partial'), fresh=msg(raw), node=el(fresh);
const root=fresh.return;
const staleHost={stateNode:node,return:old}; old.child=staleHost;
old.return=root; old.alternate=fresh; fresh.alternate=old;
node['__reactFiber$fixture']=staleHost;
if(extract(node)!==raw) throw Error('stale alternate was read');
const orphan=el(msg(raw)); orphan.isConnected=false;
if(extract(orphan)!==null) throw Error('detached accepted');
const unreachable=el(msg(raw)); unreachable['__reactFiber$fixture'].return.return.child=null;
if(extract(unreachable)!==null) throw Error('uncommitted host accepted');
'''.replace('EXTRACTOR', main.AGENT_SOURCE_EXTRACTOR)
        result = subprocess.run(['node', '-e', script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
