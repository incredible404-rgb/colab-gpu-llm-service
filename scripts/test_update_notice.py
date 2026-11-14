import contextlib
import io
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

import update_notice as un


class VersionTests(unittest.TestCase):
    def test_only_plain_versions_compare(self):
        self.assertTrue(un.newer_than('v1.2.0', 'v1.1.9'))
        self.assertTrue(un.newer_than('v1.1.10', 'v1.1.9'))  # numbers, not text
        self.assertFalse(un.newer_than('v1.1.2', 'v1.1.2'))
        self.assertFalse(un.newer_than('v1.1.1', 'v1.1.2'))
        self.assertFalse(un.newer_than('v1.2.0-beta.1', 'v1.1.2'))  # a prerelease is never announced
        self.assertFalse(un.newer_than('v1.2.0', ''))
        self.assertFalse(un.newer_than('', 'v1.1.2'))


class NoticeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = Path(self.tmp.name) / 'cfg' / 'update-check.json'
        self.asked = []
        env = {k: v for k, v in os.environ.items() if k != 'COLAB_UPDATE_CHECK'}
        for patcher in (mock.patch.object(un, 'STATE', self.state),
                        mock.patch.object(un, 'current_tag', return_value='v1.1.2'),
                        mock.patch.object(un, 'latest_tag', side_effect=self._latest),
                        mock.patch.dict(os.environ, env, clear=True)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.latest = 'v1.2.0'

    def _latest(self):
        self.asked.append(1)
        if isinstance(self.latest, Exception):
            raise self.latest
        return self.latest

    def test_a_newer_release_is_named_with_how_to_update(self):
        line = un.message(now=1_000_000)
        self.assertIn('v1.2.0 is out (this clone has v1.1.2)', line)
        self.assertIn('/releases/tag/v1.2.0', line)
        self.assertIn('pull --ff-only', line)
        self.assertEqual(json.loads(self.state.read_text(encoding='utf-8'))['latest'], 'v1.2.0')

    def test_github_is_asked_once_a_day_and_the_answer_is_kept(self):
        un.message(now=1_000_000)
        self.assertIn('v1.2.0', un.message(now=1_000_000 + 3600))
        self.assertEqual(len(self.asked), 1)
        un.message(now=1_000_000 + un.EVERY_S)
        self.assertEqual(len(self.asked), 2)

    def test_nothing_is_said_when_current_off_or_not_a_tagged_clone(self):
        self.latest = 'v1.1.2'
        self.assertEqual(un.message(now=1_000_000), '')
        self.latest = 'v1.2.0'
        with mock.patch.dict(os.environ, {'COLAB_UPDATE_CHECK': '0'}):
            self.assertEqual(un.message(now=9_000_000), '')
        with mock.patch.object(un, 'current_tag', return_value=''):
            self.assertEqual(un.message(now=9_000_000), '')
        self.assertEqual(len(self.asked), 1)  # neither of the last two asked GitHub

    def test_offline_keeps_quiet_and_waits_a_day(self):
        self.latest = OSError('no network')
        self.assertEqual(un.message(now=1_000_000), '')
        self.assertEqual(un.message(now=1_000_000 + 60), '')
        self.assertEqual(len(self.asked), 1)

    def test_show_writes_to_stderr_only_and_never_raises(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            un.show()
        self.assertEqual(out.getvalue(), '')
        self.assertIn('[update]', err.getvalue())
        with mock.patch.object(un, 'message', side_effect=RuntimeError('boom')), \
                contextlib.redirect_stderr(io.StringIO()):
            un.show()


class LatestTagTests(unittest.TestCase):
    def _serve(self, location):
        class Handler(BaseHTTPRequestHandler):
            def do_HEAD(self):
                if location:
                    self.send_response(302)
                    self.send_header('Location', location)
                else:
                    self.send_response(404)
                self.end_headers()

            def log_message(self, *args):
                pass

        server = HTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return 'http://127.0.0.1:%d/latest' % server.server_address[1]

    def test_the_redirect_names_the_tag_and_is_not_followed(self):
        url = self._serve('https://github.com/o/r/releases/tag/v9.8.7')
        with mock.patch.object(un, 'LATEST_URL', url):
            self.assertEqual(un.latest_tag(), 'v9.8.7')

    def test_no_release_is_an_empty_tag(self):
        with mock.patch.object(un, 'LATEST_URL', self._serve('')):
            self.assertEqual(un.latest_tag(), '')


if __name__ == '__main__':
    unittest.main()
