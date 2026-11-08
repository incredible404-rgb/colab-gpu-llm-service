"""Isolated tests for local_novel_mcp tools; every test runs against a throwaway NOVEL_ROOT."""
import json
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

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


if __name__ == '__main__':
    unittest.main()
