"""Isolated tests for local_novel_mcp tools; every test runs against a throwaway NOVEL_ROOT."""
import http.client
import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_novel_mcp as m  # noqa: E402

BOOK = 'book'
NOVEL = f'{BOOK}/正文.md'
REQ = f'{BOOK}/写作要求.md'


def call(name, **args):
    return m._call_tool(name, args)


class NovelToolsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='novel-mcp-test-'))
        self._orig_root = m.NOVEL_ROOT
        m.NOVEL_ROOT = self.tmp.resolve()
        (self.tmp / BOOK).mkdir()
        (self.tmp / NOVEL).write_bytes(
            '# 书名\n\n## 第一章 温明园\n貂蝉在园中。\n六胎之事。\n\n## 第二章 虎牢\n貂蝉又来。\n六胎已定。\n'.encode('utf-8'))
        (self.tmp / REQ).write_bytes('设定：六胎\r\n称谓：主公\r\n'.encode('utf-8'))
        (self.tmp / BOOK / '参考').mkdir()
        (self.tmp / BOOK / '参考' / '原版.txt').write_bytes('第一回 废天子\n貂蝉登场\n第二回 斗三英\n'.encode('utf-8'))

    def tearDown(self):
        m.NOVEL_ROOT = self._orig_root
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_tools_listed_once(self):
        names = [t['name'] for t in m.TOOLS]
        self.assertEqual(len(names), len(set(names)))
        for n in names:
            self.assertNotIn('未知工具', self._safe_call(n))

    def _safe_call(self, n):
        try:
            return m._call_tool(n, {})
        except Exception as e:  # missing args are fine; unknown tool is not
            return str(e)

    def test_outline_lines_and_hanzi(self):
        out = call('outline_novel_file', path=NOVEL)
        self.assertIn('## 第一章 温明园  第 3–6 行', out)
        self.assertIn('## 第二章 虎牢  第 7–9 行', out)
        self.assertIn('# 书名  第 1–9 行', out)
        plain = call('outline_novel_file', path=f'{BOOK}/参考/原版.txt')
        self.assertIn('第一回 废天子  第 1–2 行', plain)
        level2 = call('outline_novel_file', path='ALL')
        self.assertIn('第二章 虎牢', level2)

    def test_search_multi(self):
        out = call('grep_reference', queries=['貂蝉', '六胎'], context_lines=0)
        self.assertIn('貂蝉 3 行', out)
        self.assertIn('六胎 3 行', out)
        self.assertIn('原版.txt', out)
        lit = call('grep_reference', query='第.章', regex=False)
        self.assertIn('第.章 0 行', lit)
        glob = call('grep_reference', query='貂蝉', glob='*.txt')
        self.assertIn('共搜 1 个文件', glob)

    def test_edit_unique_match_required(self):
        before = (self.tmp / NOVEL).read_bytes()
        out = call('edit_novel_file', path=NOVEL, old_text='貂蝉', new_text='任红昌')
        self.assertIn('命中 2 处', out)
        self.assertIn('第 4, 8 行', out)
        self.assertEqual((self.tmp / NOVEL).read_bytes(), before)
        out = call('edit_novel_file', path=NOVEL, old_text='貂蝉', new_text='任红昌', replace_all=True)
        self.assertIn('替换 2 处', out)
        self.assertIn('净汉字', out)

    def test_batch_atomic_and_crlf_kept(self):
        before = (self.tmp / NOVEL).read_bytes()
        out = call('edit_novel_file', edits=[
            {'path': NOVEL, 'old_text': '六胎之事', 'new_text': '七胎之事'},
            {'path': REQ, 'old_text': '不存在的字', 'new_text': 'x'},
        ])
        self.assertIn('未执行任何修改', out)
        self.assertEqual((self.tmp / NOVEL).read_bytes(), before)
        out = call('edit_novel_file', edits=[
            {'path': NOVEL, 'old_text': '六胎', 'new_text': '七胎', 'replace_all': True},
            {'path': REQ, 'old_text': '六胎', 'new_text': '七胎'},
        ])
        self.assertEqual(out.count('已更新'), 2)
        self.assertEqual((self.tmp / REQ).read_bytes(), '设定：七胎\r\n称谓：主公\r\n'.encode('utf-8'))
        self.assertNotIn('六胎', (self.tmp / NOVEL).read_text(encoding='utf-8'))

    def test_history_and_version_restore(self):
        call('edit_novel_file', path=NOVEL, old_text='温明园', new_text='温明园（一稿）')
        time.sleep(1.1)
        call('edit_novel_file', path=NOVEL, old_text='（一稿）', new_text='（二稿）')
        hist = call('list_novel_history', path=NOVEL)
        versions = [line.split()[0].split('=')[1] for line in hist.splitlines() if line.startswith('version=')]
        self.assertEqual(len(versions), 2)
        oldest = versions[-1]
        out = call('undo_novel_write', path=NOVEL, version=oldest)
        self.assertIn('版本 ' + oldest, out)
        text = (self.tmp / NOVEL).read_text(encoding='utf-8')
        self.assertIn('## 第一章 温明园\n', text)
        self.assertIn('找不到版本', call('undo_novel_write', path=NOVEL, version='19990101-000000'))

    def test_move_rename_carry_snapshots(self):
        call('edit_novel_file', path=NOVEL, old_text='温明园', new_text='温明园改')
        out = call('move_novel_file', path=NOVEL, new_name='正文v2.md')
        self.assertIn('随行快照 1 份', out)
        new = f'{BOOK}/正文v2.md'
        self.assertIn('version=', call('list_novel_history', path=new))
        out = call('move_novel_file', path=f'{BOOK}/参考', new_path=f'{BOOK}/参考章节/参考1')
        self.assertTrue((self.tmp / BOOK / '参考章节' / '参考1' / '原版.txt').is_file())
        self.assertIn('目标已存在', call('move_novel_file', path=new, new_path=REQ))
        with self.assertRaises(ValueError):
            call('move_novel_file', path=BOOK, new_path=f'{BOOK}/inner')
        with self.assertRaises(ValueError):
            call('move_novel_file', path='archive/.snapshots', new_path='x')

    def test_delete_goes_to_trash_and_restores(self):
        out = call('delete_novel_file', path=REQ)
        self.assertFalse((self.tmp / REQ).exists())
        restore = json.loads('"' + out.split('move_novel_file(path="')[1].split('"')[0] + '"')
        self.assertTrue((self.tmp / restore).is_file())
        self.assertNotIn('写作要求', call('list_novel_files'))
        call('move_novel_file', path=restore, new_path=REQ)
        self.assertTrue((self.tmp / REQ).is_file())
        with self.assertRaises(ValueError):
            call('delete_novel_file', path='archive')

    def test_write_append_still_reports_counts(self):
        out = call('write_novel_file', path=NOVEL, content='## 第三章 受禅台\n吕凤先登台。\n')
        self.assertIn('总行数', out)
        self.assertIn('本轮净汉字', out)
        self.assertIn('第三章', call('outline_novel_file', path=NOVEL))


def _age(path, seconds=60):
    """Back-date a file so the lines cache treats it as settled (not racily clean)."""
    ns = time.time_ns() - seconds * 1_000_000_000
    os.utime(path, ns=(ns, ns))


class LinesCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='novel-cache-test-'))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _file(self, name, text, aged=True):
        p = self.tmp / name
        p.write_bytes(text.encode('utf-8') if isinstance(text, str) else text)
        if aged:
            _age(p)
        return p

    def test_hit_then_invalidated_by_size_or_mtime(self):
        cache = m._LinesCache()
        p = self._file('a.md', '一\n二\n')
        self.assertEqual(cache.get(p).lines, ('一', '二'))
        cache.get(p)
        self.assertEqual((cache.hits, cache.misses), (1, 1))
        p.write_bytes('一\n二\n三\n'.encode('utf-8'))
        _age(p, 30)
        self.assertEqual(cache.get(p).lines, ('一', '二', '三'))
        same_size = '甲\n乙\n丙\n'
        p.write_bytes(same_size.encode('utf-8'))
        _age(p, 10)
        self.assertEqual(cache.get(p).lines, ('甲', '乙', '丙'))
        self.assertEqual(cache.misses, 3)

    def test_racy_young_file_is_never_served_stale(self):
        cache = m._LinesCache()
        p = self._file('young.md', 'aaa\n', aged=False)
        self.assertEqual(cache.get(p).lines, ('aaa',))
        p.write_bytes(b'bbb\n')  # same size, same coarse timestamp possible
        self.assertEqual(cache.get(p).lines, ('bbb',))
        self.assertEqual(cache.hits, 0)

    def test_bounded_by_entries_and_cost(self):
        cache = m._LinesCache(max_cost=10_000, max_entries=3)
        paths = [self._file(f'{i}.md', f'line {i}\n') for i in range(6)]
        for p in paths:
            cache.get(p)
        self.assertLessEqual(len(cache._items), 3)
        big = self._file('big.md', 'x' * 20_000)
        cache.get(big)
        self.assertNotIn(str(big), cache._items)
        self.assertLessEqual(cache._cost, cache.max_cost)
        cache.get(paths[-1])
        self.assertGreaterEqual(cache.hits, 1)
        cache.clear()
        self.assertEqual((len(cache._items), cache._cost), (0, 0))

    def test_threads_share_one_cache(self):
        cache = m._LinesCache(max_cost=50_000, max_entries=4)
        paths = [self._file(f'{i}.md', ''.join(f'{i}-{k}\n' for k in range(50))) for i in range(8)]
        errors = []

        def work():
            try:
                for _ in range(200):
                    for i, p in enumerate(paths):
                        assert cache.get(p).lines[0] == f'{i}-0'
            except Exception as err:  # pragma: no cover - reported below
                errors.append(err)

        threads = [threading.Thread(target=work) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertLessEqual(len(cache._items), 4)
        self.assertEqual(cache._cost, sum(e.cost for _, e in cache._items.values()))

    def test_decoding_matches_text_mode_splitlines(self):
        samples = [
            '一\r\n二\r\n', '甲\r乙\r丙', 'a\x0bb\x0cc\x1cd\x1de\x1ef\x85g h i\n', '﻿BOM\n正文', '',
            '\n\n', 'x\r\r\ny\n\r', b'ok\xffbad\xc3\n\xe4\xb8',
        ]
        cache = m._LinesCache()
        for i, sample in enumerate(samples):
            p = self._file(f's{i}.md', sample)
            expect = p.read_text(encoding='utf-8', errors='replace').splitlines()
            self.assertEqual(list(cache.get(p).lines), expect, repr(sample))

    def test_hanzi_prefix_matches_section_counts(self):
        p = self._file('h.md', '# 书\n你好a\n\n世界，ok\n')
        entry = m._LinesCache().get(p)
        text = p.read_text(encoding='utf-8')
        prefix = entry.hanzi_prefix()
        self.assertEqual(prefix[-1], m._count_hanzi(text))
        for start in range(1, len(entry.lines) + 1):
            for end in range(start, len(entry.lines) + 1):
                self.assertEqual(prefix[end] - prefix[start - 1], m._count_hanzi('\n'.join(entry.lines[start - 1:end])))


class WalkEquivalenceTest(unittest.TestCase):
    """The pruned walk must list exactly what the old rglob + filter did, in the same order."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='novel-walk-test-')).resolve()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self._orig = m.NOVEL_ROOT
        m.NOVEL_ROOT = self.tmp
        self.addCleanup(setattr, m, 'NOVEL_ROOT', self._orig)
        for rel in [
            'a/x.md', 'a.md', 'a-b.md', 'A/Y.TXT', 'b/c/d.md', 'b/c/e.txt', 'b/skip.png', 'b/.hidden.md',
            '.git/config.md', 'archive/old.md', 'archive/.snapshots/b/x.md.20260101-000000.bak', 'b/archive/n.md',
            'b/archive.md', 'z/archive/q/r.md', '.snapshots/s.md', 'b/.cache/t.md', '文/章.md', 'b/c.d.md',
        ]:
            f = self.tmp / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text('内容\n', encoding='utf-8')

    def _reference(self, base, include_archive):
        out = []
        for p in sorted(base.rglob('*')):
            if not p.is_file() or p.suffix.lower() not in ('.md', '.txt'):
                continue
            parts = p.relative_to(m.NOVEL_ROOT).parts
            if any(part.startswith('.') for part in parts):
                continue
            if not include_archive and 'archive' in parts:
                continue
            out.append(p)
        return out

    def test_matches_rglob_filter_for_every_directory(self):
        dirs = [self.tmp] + [p for p in self.tmp.rglob('*') if p.is_dir()]
        for base in dirs:
            for include_archive in (False, True):
                self.assertEqual(m._md_files(base, include_archive), self._reference(base, include_archive), (base, include_archive))

    def test_list_tool_matches_old_filter(self):
        out = call('list_novel_files')
        expect = [p.relative_to(self.tmp).as_posix() for p in self._reference(self.tmp, False)]
        self.assertEqual([line.split('  (')[0] for line in out.splitlines()], expect)
        self.assertIn('(1 行, ', out.splitlines()[0])
        self.assertEqual(call('list_novel_files', folder='archive'), '未找到 .md / .txt 文件')
        self.assertEqual(call('list_novel_files', folder='a.md'), '未找到 .md / .txt 文件')

    def test_grep_dedupes_overlapping_paths_in_order(self):
        out = call('grep_reference', query='内容', paths=['b', 'b/c', 'b/c/d.md'], max_matches=60)
        self.assertIn('共搜 %d 个文件' % len(self._reference(self.tmp / 'b', False)), out)


class AuditIndexTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='novel-audit-test-')).resolve()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.audit = self.tmp / 'write_audit.jsonl'

    def _row(self, path, status='ok', tool='write_novel_file', **extra):
        return json.dumps(dict({'tool': tool, 'path': path, 'status': status, 'mode': 'append'}, **extra), ensure_ascii=False) + '\n'

    def _full(self, rel):
        rows = []
        for line in self.audit.read_text(encoding='utf-8').splitlines():
            try:
                rec = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            if rec.get('tool') == 'write_novel_file' and rec.get('status') == 'ok' \
                    and str(rec.get('path') or '').replace('\\', '/').lstrip('/') == rel:
                rows.append(rec)
        return rows

    def test_incremental_equals_full_parse(self):
        index = m._AuditIndex()
        self.audit.write_text(self._row('b/x.md', n=1) + self._row('b/y.md') + 'not json\n' + self._row('b\\x.md', status='rejected'),
                              encoding='utf-8', newline='')
        self.assertEqual(index.rows_for(self.audit, 'b/x.md'), self._full('b/x.md'))
        with self.audit.open('ab') as fh:
            fh.write(self._row('/b/x.md', n=2).encode('utf-8'))
            fh.write(self._row('c.md', tool='edit_novel_file').encode('utf-8'))
        got = index.rows_for(self.audit, 'b/x.md')
        self.assertEqual(got, self._full('b/x.md'))
        self.assertEqual([r['n'] for r in got], [1, 2])
        self.assertEqual(index.rows_for(self.audit, 'c.md'), [])
        self.assertEqual(index.offset, self.audit.stat().st_size)

    def test_torn_last_line_is_seen_but_not_committed(self):
        index = m._AuditIndex()
        whole = self._row('b/x.md', n=1)
        full_row = self._row('b/x.md', n=2)
        self.audit.write_bytes((whole + full_row[:-1]).encode('utf-8'))  # last row lacks its newline
        self.assertEqual([r['n'] for r in index.rows_for(self.audit, 'b/x.md')], [1, 2])
        self.assertEqual(index.offset, len(whole.encode('utf-8')))
        with self.audit.open('ab') as fh:
            fh.write(b'\n')
        self.assertEqual([r['n'] for r in index.rows_for(self.audit, 'b/x.md')], [1, 2])
        self.assertEqual(index.offset, self.audit.stat().st_size)
        self.audit.write_bytes((whole + '{"tool": "write_novel_file", "pa').encode('utf-8'))  # shrank + new torn tail
        self.assertEqual([r['n'] for r in index.rows_for(self.audit, 'b/x.md')], [1])

    def test_rewrite_or_other_file_rebuilds(self):
        index = m._AuditIndex()
        self.audit.write_text(self._row('b/x.md', n=1) + self._row('b/x.md', n=2), encoding='utf-8', newline='')
        self.assertEqual(len(index.rows_for(self.audit, 'b/x.md')), 2)
        # different bytes under the already-parsed prefix, then growth: the tail check must notice the rewrite
        self.audit.write_text(self._row('b/x.md', n=7) + self._row('b/x.md', n=8) + self._row('b/x.md', n=9),
                              encoding='utf-8', newline='')
        self.assertEqual([r['n'] for r in index.rows_for(self.audit, 'b/x.md')], [7, 8, 9])
        other = self.tmp / 'other.jsonl'
        other.write_text(self._row('b/x.md', n=5), encoding='utf-8', newline='')
        self.assertEqual([r['n'] for r in index.rows_for(other, 'b/x.md')], [5])

    def test_bad_utf8_raises_and_does_not_double_count_later(self):
        index = m._AuditIndex()
        self.audit.write_bytes(self._row('b/x.md', n=1).encode('utf-8') + b'\xff\xfe\n')
        with self.assertRaises(UnicodeDecodeError):
            index.rows_for(self.audit, 'b/x.md')
        self.audit.write_bytes(self._row('b/x.md', n=1).encode('utf-8'))
        self.assertEqual([r['n'] for r in index.rows_for(self.audit, 'b/x.md')], [1])


class HttpKeepAliveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = m.ThreadingHTTPServer(('127.0.0.1', 0), m.MCPHandler)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _post(self, conn, payload):
        conn.request('POST', '/mcp', body=json.dumps(payload), headers={'Content-Type': 'application/json'})
        return conn.getresponse()

    def test_many_requests_share_one_connection(self):
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        conn.connect()
        sock = conn.sock
        for i in range(20):
            resp = self._post(conn, {'jsonrpc': '2.0', 'id': i, 'method': 'ping'})
            self.assertEqual(resp.version, 11)
            self.assertEqual(json.loads(resp.read()), {'jsonrpc': '2.0', 'id': i, 'result': {}})
            self.assertIs(conn.sock, sock)
        resp = self._post(conn, {'jsonrpc': '2.0', 'method': 'notifications/initialized'})
        self.assertEqual((resp.status, resp.read()), (202, b''))
        conn.request('GET', '/health')
        resp = conn.getresponse()
        self.assertTrue(json.loads(resp.read())['ok'])
        conn.request('OPTIONS', '/mcp', headers={'Origin': 'http://x', 'Access-Control-Request-Headers': 'x-a'})
        resp = conn.getresponse()
        self.assertEqual(resp.status, 204)
        self.assertEqual(resp.getheader('Access-Control-Allow-Private-Network'), 'true')
        resp.read()
        conn.request('DELETE', '/mcp')
        resp = conn.getresponse()
        self.assertEqual((resp.status, resp.read()), (200, b''))
        conn.request('GET', '/nope')
        resp = conn.getresponse()
        self.assertEqual((resp.status, resp.read()), (405, b''))
        self.assertIs(conn.sock, sock)
        conn.close()

    def test_http10_client_and_connection_close_still_work(self):
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        conn.request('POST', '/mcp', body=json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/list'}),
                     headers={'Connection': 'close'})
        resp = conn.getresponse()
        self.assertEqual(len(json.loads(resp.read())['result']['tools']), len(m.TOOLS))
        conn.close()
        with socket.create_connection(('127.0.0.1', self.port), timeout=10) as raw:
            raw.sendall(b'GET /health HTTP/1.0\r\n\r\n')
            data = b''
            while chunk := raw.recv(4096):
                data += chunk
        self.assertTrue(data.startswith(b'HTTP/1.0 200') or data.startswith(b'HTTP/1.1 200'))
        self.assertIn(b'"ok": true', data)

    def test_bad_json_keeps_the_stream_usable(self):
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        conn.request('POST', '/mcp', body='{nope')
        resp = conn.getresponse()
        self.assertEqual(resp.status, 400)
        resp.read()
        resp = self._post(conn, {'jsonrpc': '2.0', 'id': 2, 'method': 'ping'})
        self.assertEqual(resp.status, 200)
        resp.read()
        conn.close()


class SkillMapTest(unittest.TestCase):
    def test_default_is_empty(self):
        with mock.patch.dict(os.environ, {}, clear=False), mock.patch.object(m, 'optional_setting', return_value=''):
            os.environ.pop('NOVEL_SKILL_MAP', None)
            self.assertEqual(m._skill_map(), {})

    def test_parse_entries(self):
        raw = 'retro=~/s/retro/SKILL.md; skill:style = /abs/style.md\nbad-entry;=nopath;noval=;ok=a=b'
        got = m._parse_skill_map(raw)
        self.assertEqual(set(got), {'skill:retro', 'skill:style', 'skill:ok'})
        self.assertEqual(got['skill:retro'], Path.home() / 's/retro/SKILL.md')
        self.assertEqual(got['skill:style'], Path('/abs/style.md'))
        self.assertEqual(got['skill:ok'], Path('a=b'))

    def test_secrets_value_used_and_env_wins(self):
        with mock.patch.object(m, 'optional_setting', return_value='one=/p/one.md'):
            with mock.patch.dict(os.environ, {'NOVEL_SKILL_MAP': 'two=/p/two.md'}):
                self.assertEqual(set(m._skill_map()), {'skill:two'})
            with mock.patch.dict(os.environ, {}):
                os.environ.pop('NOVEL_SKILL_MAP', None)
                self.assertEqual(set(m._skill_map()), {'skill:one'})

    def test_safe_path_resolves_configured_skill_only(self):
        with mock.patch.object(m, 'SKILL_MAP', {'skill:x': Path('/p/x.md')}):
            self.assertEqual(m._safe_path('skill:x'), Path('/p/x.md'))
            # An unconfigured skill: name is just a file name under the novel root, never a path elsewhere.
            self.assertEqual(m._safe_path('skill:unknown').parent, m.NOVEL_ROOT)

    def test_description_lists_only_configured_names(self):
        desc = [t for t in m.TOOLS if t['name'] == 'read_novel_file'][0]['description']
        for name in m.SKILL_MAP:
            self.assertIn(name, desc)
        if not m.SKILL_MAP:
            self.assertNotIn('skill:', desc)


if __name__ == '__main__':
    unittest.main()
