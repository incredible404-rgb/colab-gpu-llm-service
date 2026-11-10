"""bench_ab end to end against a fake llama-server (no Colab, no network)."""
import json
import re
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import bench_ab


class Fake(BaseHTTPRequestHandler):
    drop_recall = False

    def log_message(self, *a):
        pass

    def _send(self, obj):
        raw = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self._send({'data': [{'id': 'fake-model'}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.path == '/tokenize':
            return self._send({'tokens': body['content'].split()})
        text = body['messages'][0]['content']
        msg = {'content': 'ok', 'reasoning_content': 'r'}
        if 'tools' in body:
            city = re.search(r'查一下 (\S+)', text).group(1)
            msg = {'content': '', 'tool_calls': [{'function': {
                'name': 'get_weather', 'arguments': json.dumps({'city': city, 'unit': 'celsius'})}}]}
        m = re.search(r'passphrase is (\S+)\.', text)
        if m:
            msg = {'content': '' if Fake.drop_recall else m.group(1)}
        self._send({'choices': [{'message': msg, 'finish_reason': 'stop'}],
                    'usage': {'completion_tokens': 256},
                    'timings': {'predicted_per_second': 100.0, 'prompt_per_second': 3000.0,
                                'draft_n': 30, 'draft_n_accepted': 15}})


class BenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(('127.0.0.1', 0), Fake)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.url = 'http://127.0.0.1:%d/v1' % cls.srv.server_port

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def run_cli(self, *argv):
        with tempfile.TemporaryDirectory() as d:
            code = bench_ab.main([*argv, '--base-url', self.url, '--out-dir', d])
            return code, [json.loads(p.read_text(encoding='utf-8')) for p in Path(d).glob('*.json')]

    def test_all_passes_and_records(self):
        Fake.drop_recall = False
        code, (res,) = self.run_cli('all', '--label', 't', '--lengths', '500,2000', '--tool-rounds', '6')
        self.assertEqual(code, 0)
        self.assertEqual(res['decode']['median_tps'], 100.0)
        self.assertEqual(res['decode']['draft_acceptance'], 0.5)
        self.assertEqual(res['decode']['stopped_early'], [])
        self.assertTrue(all(r['passed'] for r in res['recall']))
        self.assertEqual(res['tools']['failed'], 0)

    def test_failed_recall_gives_nonzero_exit(self):
        Fake.drop_recall = True
        try:
            code, (res,) = self.run_cli('recall', '--lengths', '500')
        finally:
            Fake.drop_recall = False
        self.assertEqual(code, 1)
        self.assertFalse(res['recall'][0]['passed'])

    def test_check_tool_call_reasons(self):
        mk = lambda m: {'choices': [{'message': m}]}
        self.assertEqual(bench_ab.check_tool_call(mk({}), 'Oslo'), 'no_tool_call')
        bad = mk({'tool_calls': [{'function': {'name': 'get_weather', 'arguments': '{oops'}}]})
        self.assertEqual(bench_ab.check_tool_call(bad, 'Oslo'), 'bad_json')
        wrong = mk({'tool_calls': [{'function': {'name': 'get_weather', 'arguments': '{"city":"Lima","unit":"celsius"}'}}]})
        self.assertEqual(bench_ab.check_tool_call(wrong, 'Oslo'), 'wrong_city')

    def test_compare_and_prefix_ratio(self):
        self.assertEqual(bench_ab.common_prefix_ratio('abcd', 'abXd'), 0.5)
        item = {'tps': 10.0, 'text': 'abc', 'completion_tokens': 256, 'draft_n': 0, 'draft_accepted': 0}
        a = {'label': 'A', 'decode': bench_ab.summarize_decode([item])}
        out = bench_ab.compare(a, a)
        self.assertIn('identical 1/1', out)
        self.assertIn('+0.0%', out)


if __name__ == '__main__':
    unittest.main()
