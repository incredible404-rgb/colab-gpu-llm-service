"""Zero-overhead local MCP server (Streamable HTTP) for llama-server WebUI novel writing.

Exposes C:\\Users\\zym\\Documents\\Colab创作 to the WebUI at http://qwen.<tailnet>.ts.net:443/ directly from the
local Edge browser via http://127.0.0.1:3426/mcp (useProxy: false), with CORS + Chromium Private
Network Access headers so the browser connects directly to localhost without going through Colab.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOST = '0.0.0.0'
PORT = 3426
NOVEL_ROOT = Path('C:/Users/zym/Documents/Colab创作').resolve()
SKILL_MAP = {
    'skill:grilling': Path('C:/Users/zym/.agents/upstream/mattpocock-skills/skills/productivity/grilling/SKILL.md'),
    'skill:prompting': Path('C:/Users/zym/.agents/skills/universal-model-prompting/SKILL.md'),
    'skill:retro': Path('C:/Users/zym/.agents/skills/retro/SKILL.md'),
}
SESSION_ID = 'local-novel-mcp-v1'


def _safe_path(rel_path: str) -> Path:
    cleaned = rel_path.strip().replace('\\', '/').lstrip('/')
    if cleaned in SKILL_MAP:
        return SKILL_MAP[cleaned]
    target = (NOVEL_ROOT / cleaned).resolve()
    if NOVEL_ROOT not in target.parents and target != NOVEL_ROOT:
        raise ValueError(f'路径超出允许的小说根目录 ({NOVEL_ROOT}): {rel_path}')
    return target


SNAPSHOT_KEEP = 30
_BAK_NAME = re.compile(r'^(?P<stem>.+)\.(?P<stamp>\d{8}-\d{6}(?:-\d+)?)\.bak$')
_CHAPTER_RE = re.compile(
    r'^\s*#{1,4}\s*第\s*([0-9一二三四五六七八九十百千两零〇]+)\s*[章节回部卷]',
    re.MULTILINE,
)
_HANZI_RE = re.compile(r'[\u4e00-\u9fff]')
# Whole-line pause markers. The next append strips the line so the half-chapter joins the prose.
_PAUSE_LINE_RE = re.compile(
    r'(?:【本章未完[^】]*】|（未完待续[^）]*）|<!--\s*未完[^>]*-->)'
)
_CN_DIGIT = {
    '零': 0, '〇': 0,
    '一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
    '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
}
_CN_UNIT = {'十': 10, '百': 100, '千': 1000}


def _parse_chapter_num(token: str) -> int | None:
    """Parse a chapter index. Arabic digits pass through; 五 / 十五 become 5 / 15."""
    token = token.strip()
    if not token:
        return None
    if token.isdigit():
        return int(token)
    total = 0
    current = 0
    seen = False
    for ch in token:
        if ch in _CN_DIGIT:
            current = _CN_DIGIT[ch]
            seen = True
        elif ch in _CN_UNIT:
            unit = _CN_UNIT[ch]
            if current == 0:
                current = 1
            total += current * unit
            current = 0
            seen = True
        else:
            return None
    if not seen:
        return None
    return total + current


def _iter_chapters(text: str) -> list[tuple[int, int, str]]:
    """Chapter headings in order: (number, line_start, heading line).

    Leading whitespace in the matcher may include the blank line before a heading.
    The cut is the heading line itself, so every byte before that line stays put.
    """
    found: list[tuple[int, int, str]] = []
    for match in _CHAPTER_RE.finditer(text):
        num = _parse_chapter_num(match.group(1))
        if num is None:
            continue
        rel_hash = match.group(0).find('#')
        hash_at = match.start() + (rel_hash if rel_hash >= 0 else 0)
        line_start = text.rfind('\n', 0, hash_at) + 1
        line_end = text.find('\n', hash_at)
        heading = text[line_start:] if line_end < 0 else text[line_start:line_end]
        found.append((num, line_start, heading.strip()))
    return found


def _opening_chapter(content: str) -> tuple[int | None, int]:
    """Return (number if content opens with a heading, total heading count)."""
    total = 0
    for match in _CHAPTER_RE.finditer(content):
        if _parse_chapter_num(match.group(1)) is not None:
            total += 1
    body = content.lstrip('\r\n\ufeff')
    opening_match = _CHAPTER_RE.match(body)
    opening = _parse_chapter_num(opening_match.group(1)) if opening_match else None
    return opening, total


def _last_chapter_title(text: str) -> str:
    chapters = _iter_chapters(text)
    return chapters[-1][2] if chapters else '（无章节标题）'


def _snapshot_dir_for(p: Path) -> Path | None:
    try:
        rel_parent = p.resolve().parent.relative_to(NOVEL_ROOT)
    except ValueError:
        return None
    return NOVEL_ROOT / 'archive' / '.snapshots' / rel_parent


def _bak_sort_key(path: Path) -> tuple:
    match = _BAK_NAME.match(path.name)
    stamp = match.group('stamp') if match else ''
    parsed = re.match(r'(\d{8}-\d{6})(?:-(\d+))?$', stamp)
    if not parsed:
        return ('', 0, path.name)
    return (parsed.group(1), int(parsed.group(2) or 1), path.name)


def _list_backups(p: Path) -> list[Path]:
    snap_dir = _snapshot_dir_for(p)
    if snap_dir is None or not snap_dir.is_dir():
        return []
    existing = [
        item for item in snap_dir.glob(p.name + '.*.bak')
        if _BAK_NAME.match(item.name) and item.name.startswith(p.name + '.')
    ]
    existing.sort(key=_bak_sort_key)
    return existing


def _backup_before_write(p: Path) -> str | None:
    """Copy an existing non-empty file under archive/.snapshots before mutation.

    Skips the copy when the newest .bak is byte-identical. Keeps at most 30 backups
    per target file. Returns the backup path relative to NOVEL_ROOT, or None when
    there is nothing to back up.
    """
    if not p.exists() or not p.is_file() or p.stat().st_size <= 0:
        return None
    snap_dir = _snapshot_dir_for(p)
    if snap_dir is None:
        return None
    snap_dir.mkdir(parents=True, exist_ok=True)
    data = p.read_bytes()
    existing = _list_backups(p)
    if existing and existing[-1].read_bytes() == data:
        return existing[-1].relative_to(NOVEL_ROOT).as_posix()
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    dest = snap_dir / f'{p.name}.{ts}.bak'
    n = 2
    while dest.exists():
        dest = snap_dir / f'{p.name}.{ts}-{n}.bak'
        n += 1
    dest.write_bytes(data)
    existing.append(dest)
    existing.sort(key=_bak_sort_key)
    while len(existing) > SNAPSHOT_KEEP:
        oldest = existing.pop(0)
        try:
            oldest.unlink()
        except OSError:
            pass
    return dest.relative_to(NOVEL_ROOT).as_posix()


def _audit_write(tool: str, path: str, mode: str, old_bytes: int, new_bytes: int, status: str, backup: str | None) -> None:
    audit = NOVEL_ROOT / 'archive' / '.snapshots' / 'write_audit.jsonl'
    audit.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        'ts': datetime.now().isoformat(timespec='seconds'),
        'tool': tool,
        'path': path,
        'mode': mode,
        'old_bytes': old_bytes,
        'new_bytes': new_bytes,
        'status': status,
        'backup': backup,
    }
    with audit.open('a', encoding='utf-8', newline='\n') as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + '\n')


TOOLS = [
    {
        'name': 'list_novel_files',
        'title': '列出本机小说目录与文件',
        'description': '列出 C:/Users/zym/Documents/Colab创作 下的书籍子目录与 .md 文件（含行数和字节大小）。',
        'inputSchema': {
            'type': 'object',
            'properties': {
                'folder': {
                    'type': 'string',
                    'description': '可选子目录名（如 "小说1_吕凤先"），留空列出全部。',
                },
            },
        },
    },
    {
        'name': 'read_novel_file',
        'title': '读取小说或写作要求文件',
        'description': '读取小说目录内的 .md 文件（如 "小说1_吕凤先/写作要求.md"），支持 start_line/end_line 分段读取，或传 tail_lines=150 一步读取正文末尾最后 N 行；也支持传 "skill:grilling"、"skill:prompting"、"skill:retro" 按需读取技能。',
        'inputSchema': {
            'type': 'object',
            'properties': {
                'path': {
                    'type': 'string',
                    'description': '相对路径（如 "小说1_吕凤先/写作要求.md" 或 "小说1_吕凤先/性转吕凤先传（重写版）.md" 或 "skill:grilling"）',
                },
                'start_line': {'type': 'integer', 'description': '起始行号（从 1 开始，可选）'},
                'end_line': {'type': 'integer', 'description': '结束行号（含，可选）'},
                'tail_lines': {'type': 'integer', 'description': '直接读取文件末尾最后 N 行（如 150，可选）'},
            },
            'required': ['path'],
        },
    },
    {
        'name': 'grep_reference',
        'title': '检索原作参考片段',
        'description': '在原作参考文件（如 "参考1_原版.md" 或 "小说1_吕凤先/吕凤先传1~4章原版参考.md"）中按关键词或正则搜索片段及上下文，避免整本读取。',
        'inputSchema': {
            'type': 'object',
            'properties': {
                'path': {'type': 'string', 'description': '要检索的文件或子目录相对路径'},
                'query': {'type': 'string', 'description': '关键词或正则表达式'},
                'context_lines': {'type': 'integer', 'description': '前后上下文行数（默认 3）'},
                'max_matches': {'type': 'integer', 'description': '最大返回命中数（默认 10）'},
            },
            'required': ['path', 'query'],
        },
    },
    {
        'name': 'edit_novel_file',
        'title': '原位修改小说或写作要求',
        'description': '将目标 .md 文件中的 old_text 精确替换为 new_text（保持 UTF-8 编码）。',
        'inputSchema': {
            'type': 'object',
            'properties': {
                'path': {'type': 'string', 'description': '文件相对路径（如 "小说1_吕凤先/写作要求.md"）'},
                'old_text': {'type': 'string', 'description': '要被替换的原文片段'},
                'new_text': {'type': 'string', 'description': '替换后的新文本'},
            },
            'required': ['path', 'old_text', 'new_text'],
        },
    },
    {
        'name': 'write_novel_file',
        'title': '追加或打回重写小说正文',
        'description': '写正文。新章或下半章用 append（默认）；只重写上一轮写入的内容用 replace_last_write；重写最后一整章用 replace_last_chapter。重复章节和误覆盖由工具自动拦截。',
        'inputSchema': {
            'type': 'object',
            'properties': {
                'path': {'type': 'string', 'description': '文件相对路径（如 "小说1_吕凤先/性转吕凤先传（重写版）.md"）'},
                'content': {'type': 'string', 'description': '要追加的新章或下半章；整章重写需以章节标题开头；只重写上一轮时不要带章节标题'},
                'mode': {
                    'type': 'string',
                    'enum': ['append', 'replace_last_chapter', 'replace_last_write'],
                    'description': '默认 append。',
                },
            },
            'required': ['path', 'content'],
        },
    },
    {
        'name': 'undo_novel_write',
        'title': '撤回上一版正文',
        'description': '用户要撤回刚才的修改时，把文件恢复到上一版。',
        'inputSchema': {
            'type': 'object',
            'properties': {
                'path': {'type': 'string', 'description': '文件相对路径（如 "小说1_吕凤先/性转吕凤先传（重写版）.md"）'},
            },
            'required': ['path'],
        },
    },
]


def _count_hanzi(text: str) -> int:
    """Count CJK unified ideographs (U+4E00–U+9FFF). Punctuation, spaces, and Markdown are not counted."""
    return len(_HANZI_RE.findall(text or ''))


def _hanzi_receipt(content: str, full_text: str) -> str:
    """Suffix for a successful write: this round's hanzi against the default 8000±500 band."""
    hanzi = _count_hanzi(content)
    total_hanzi = _count_hanzi(full_text)
    lines = len(full_text.splitlines())
    if 7500 <= hanzi <= 8500:
        band = '符合默认 8000±500 字区间'
    else:
        band = '偏离默认 7500~8500 字区间，若非用户特批或章节收尾，下轮请注意按 4 个场景节拍控量'
    return f'（本轮净汉字: {hanzi} 字 [{band}]，当前文件总汉字: {total_hanzi} 字，总行数: {lines}）'


def _strip_pause_tail(text: str) -> str:
    """Drop a trailing pause-marker line so the next append joins the prose above it.

    The whole last non-empty line must be one of: 【本章未完…】、（未完待续…）、<!-- 未完 -->.
    """
    if not text:
        return text
    trimmed = text.rstrip('\r\n')
    if not trimmed:
        return text
    cut = max(trimmed.rfind('\n'), trimmed.rfind('\r'))
    last = trimmed[cut + 1:]
    if _PAUSE_LINE_RE.fullmatch(last.strip()) is None:
        return text
    prefix = trimmed[:cut + 1] if cut >= 0 else ''
    prose = prefix.rstrip('\r\n')
    if not prose:
        return ''
    nl = '\r\n' if '\r\n' in text else '\n'
    return prose + nl + nl


def _ok_novel_writes(rel: str) -> list[dict]:
    audit = NOVEL_ROOT / 'archive' / '.snapshots' / 'write_audit.jsonl'
    if not audit.is_file():
        return []
    rel_posix = rel.replace('\\', '/').lstrip('/')
    rows: list[dict] = []
    with audit.open('r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            path = str(rec.get('path') or '').replace('\\', '/').lstrip('/')
            if rec.get('tool') != 'write_novel_file' or rec.get('status') != 'ok':
                continue
            if path != rel_posix:
                continue
            rows.append(rec)
    return rows


def _resolve_backup_file(backup_rel: str | None) -> Path | None:
    if not backup_rel:
        return None
    bak = (NOVEL_ROOT / str(backup_rel)).resolve()
    if NOVEL_ROOT not in bak.parents and bak != NOVEL_ROOT:
        return None
    if bak.is_file():
        return bak
    return None


def _last_append_has_backup(rel: str) -> bool:
    """Latest successful write was an append and its pre-image snapshot is still on disk."""
    rows = _ok_novel_writes(rel)
    if not rows:
        return False
    last = rows[-1]
    return last.get('mode') == 'append' and _resolve_backup_file(last.get('backup')) is not None


def _record_is_append(rec: dict) -> bool:
    return rec.get('mode') == 'append' or rec.get('action') == 'append'


def _preimage_bytes(rec: dict, p: Path) -> tuple[bytes, str] | None:
    """File bytes from before this audit row.

    The backup snapshot is the pre-image. If that file is gone, `old_bytes` is the
    prefix length of the file as it stood before the write.
    """
    bak = _resolve_backup_file(rec.get('backup'))
    if bak is not None:
        return bak.read_bytes(), bak.relative_to(NOVEL_ROOT).as_posix()
    old_n = int(rec.get('old_bytes') or 0)
    if old_n == 0 and not rec.get('backup'):
        return b'', '上一次追加前文件为空'
    if old_n > 0:
        current = p.read_bytes() if p.is_file() else b''
        if old_n <= len(current):
            return current[:old_n], f'最近一次追加前的 old_bytes={old_n} 前缀'
    return None


def _base_before_last_write(rel: str, p: Path) -> tuple[bytes, str]:
    """Bytes from before the second half was appended.

    The anchor is the latest successful append (mode or action), specifically that
    row's backup / old_bytes — the file after the first half and before the second
    half. Later replace_last_write snapshots still contain the rejected half and
    must not become the base, even across five rewrites in a row. A full-chapter
    replace that sits after the append is also not the base: restoring it and
    appending would put the old chapter back.
    """
    rows = _ok_novel_writes(rel)
    for rec in reversed(rows):
        if not _record_is_append(rec):
            continue
        found = _preimage_bytes(rec, p)
        if found is not None:
            return found
        break
    if rows:
        idx = len(rows) - 1
        if rows[-1].get('mode') == 'replace_last_write':
            while idx >= 0 and rows[idx].get('mode') == 'replace_last_write':
                idx -= 1
        if idx >= 0 and not _record_is_append(rows[idx]) and rows[idx].get('mode') != 'replace_last_write':
            found = _preimage_bytes(rows[idx], p)
            if found is not None:
                return found
    current = p.read_bytes() if p.is_file() else b''
    for bak in reversed(_list_backups(p)):
        data = bak.read_bytes()
        if data != current:
            return data, bak.relative_to(NOVEL_ROOT).as_posix()
    raise ValueError('没有可回退的上一次写入快照（write_audit.jsonl 与 .bak 都不可用）')


def _cut_same_chapter(raw: bytes, text: str, content: str, chapter_no: int) -> bytes | None:
    """Swap the trailing chapter when it is `chapter_no`; None if the tail is another chapter.

    replace_last_write uses this so a titled chapter is not appended onto a snapshot
    that already ends with that same chapter.
    """
    chapters = _iter_chapters(text)
    if len(chapters) < 2 or chapters[-1][0] != chapter_no:
        return None
    return _replace_last_chapter_bytes(raw, text, content)


def _append_content_bytes(raw: bytes, text: str, content: str) -> bytes | None:
    """Append content after the existing UTF-8 bytes. None means the tail is already present.

    A trailing pause line is removed first so a half-chapter joins the prose above the marker.
    """
    if text.encode('utf-8') != raw:
        raise ValueError('原文 UTF-8 往返不一致，已中止写入以免改动前序章节')
    text = _strip_pause_tail(text)
    stripped = content.strip()
    if len(stripped) > 100 and text.rstrip().endswith(stripped):
        return None
    sep = ''
    if text and not text.endswith('\n\n'):
        sep = '\n' if text.endswith('\n') else '\n\n'
    updated = text + sep + content.lstrip('\n')
    if not updated.endswith('\n'):
        updated += '\n'
    return updated.encode('utf-8')


def _replace_last_chapter_bytes(raw: bytes, text: str, content: str) -> bytes:
    """Keep every byte before the last chapter heading and swap that chapter for content."""
    chapters = _iter_chapters(text)
    if len(chapters) < 2:
        raise ValueError('没有可锁定的前序章节')
    start = chapters[-1][1]
    prefix = text[:start].encode('utf-8')
    if raw[:len(prefix)] != prefix:
        raise ValueError('无法按字节保留前序章节，已中止写入')
    body = content[1:] if content.startswith('\ufeff') else content
    body = body.lstrip('\r\n')
    if not body.strip():
        raise ValueError('content 为空')
    if not body.endswith('\n'):
        body += '\n'
    return prefix + body.encode('utf-8')


def _guarded_write(rel: str, p: Path, mode: str, content: str) -> str:
    """Route append / replace_last_write / replace_last_chapter for an existing novel.

    Same-number single-chapter content replaces only the last chapter.
    A heading-less replace_last_chapter after an append downgrades to replace_last_write.
    overwrite of chapter N when the file ends at N-1 is forced to append.
    Any other overwrite is rejected and does not touch the file.
    """
    raw = p.read_bytes()
    old_bytes = len(raw)
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError:
        _audit_write('write_novel_file', rel, mode, old_bytes, old_bytes, 'rejected', None)
        return f'拒绝写入：{rel} 不是 UTF-8 文本（{old_bytes} 字节），已中止以免损坏前序章节。'

    chapters = _iter_chapters(text)
    opening, heading_count = _opening_chapter(content)
    last = chapters[-1] if chapters else None
    has_prior_chapters = len(chapters) >= 2

    def _reject_overwrite() -> str:
        existing_lines = len(text.splitlines())
        _audit_write('write_novel_file', rel, mode, old_bytes, old_bytes, 'rejected', None)
        return (
            f'拒绝覆写：{rel} 已存在（{old_bytes} 字节 / {existing_lines} 行）。'
            '为防止前文丢失，write_novel_file 禁止对已有文件使用 overwrite 模式！'
            '若要续写新章节，请传 mode="append"（或不传 mode）；'
            '若要只重写上一轮追加的半章，请传 mode="replace_last_write"；'
            '若要打回重写最后一章，请传 mode="replace_last_chapter"（只替换末章）；'
            '若要修改已有段落，请使用 edit_novel_file。'
        )

    def _do_append(auto_note: str | None) -> str:
        try:
            updated = _append_content_bytes(raw, text, content)
        except ValueError as err:
            _audit_write('write_novel_file', rel, 'append', old_bytes, old_bytes, 'rejected', None)
            return f'拒绝写入：{err}'
        if updated is None:
            _audit_write('write_novel_file', rel, 'append', old_bytes, old_bytes, 'duplicate', None)
            return f'未重复追加：{rel} 末尾已包含这段正文，无需重复追加。'
        backup = _backup_before_write(p)
        p.write_bytes(updated)
        new_bytes = p.stat().st_size
        _audit_write('write_novel_file', rel, 'append', old_bytes, new_bytes, 'ok', backup)
        full = updated.decode('utf-8')
        lines = len(full.splitlines())
        if auto_note:
            msg = f'{auto_note}（前序章节完整保留，旧版已快照至 {backup}）。当前 {new_bytes} 字节 / {lines} 行。'
        else:
            msg = f'已成功追加 {len(content)} 字符到 {rel}（当前总行数: {lines}）'
        return msg + _hanzi_receipt(content, full)

    def _do_replace(auto: bool, chapter_no: int) -> str:
        try:
            updated = _replace_last_chapter_bytes(raw, text, content)
        except ValueError as err:
            _audit_write('write_novel_file', rel, 'replace_last_chapter', old_bytes, old_bytes, 'rejected', None)
            return f'拒绝替换：{err}'
        backup = _backup_before_write(p)
        p.write_bytes(updated)
        new_bytes = p.stat().st_size
        new_text = updated.decode('utf-8')
        _audit_write('write_novel_file', rel, 'replace_last_chapter', old_bytes, new_bytes, 'ok', backup)
        title = _last_chapter_title(new_text)
        lines = len(new_text.splitlines())
        verb = '已自动替换' if auto else '已替换'
        return (
            f'{verb}文件末尾的第 {chapter_no} 章（前序章节完整保留，旧版已快照至 {backup}）。'
            f'当前 {new_bytes} 字节 / {lines} 行，末章「{title}」。'
        ) + _hanzi_receipt(content, new_text)

    def _do_replace_last_write(auto_note: str | None) -> str:
        try:
            base_raw, base_note = _base_before_last_write(rel, p)
            base_text = base_raw.decode('utf-8')
            updated = None
            cut_note = None
            # A titled payload for the chapter already at the tail must cut that
            # chapter off. Appending it onto the restored snapshot (or the live
            # file) would leave two copies of the same chapter.
            if opening is not None:
                updated = _cut_same_chapter(base_raw, base_text, content, opening)
                if updated is not None:
                    cut_note = '同章标题，已切掉基底末尾旧章再写入'
                else:
                    updated = _cut_same_chapter(raw, text, content, opening)
                    if updated is not None:
                        cut_note = '同章标题，已切掉当前文件末尾旧章再写入'
                        base_note = '当前文件'
            if updated is None:
                updated = _append_content_bytes(base_raw, base_text, content)
        except (ValueError, UnicodeDecodeError) as err:
            _audit_write('write_novel_file', rel, 'replace_last_write', old_bytes, old_bytes, 'rejected', None)
            return f'拒绝替换上一次写入：{err}'
        if updated is None:
            _audit_write('write_novel_file', rel, 'replace_last_write', old_bytes, old_bytes, 'duplicate', None)
            return f'未重复写入：{rel} 末尾已包含这段正文，无需重写上一轮。'
        backup = _backup_before_write(p)
        p.write_bytes(updated)
        new_bytes = p.stat().st_size
        _audit_write('write_novel_file', rel, 'replace_last_write', old_bytes, new_bytes, 'ok', backup)
        full = updated.decode('utf-8')
        lines = len(full.splitlines())
        if auto_note:
            msg = f'{auto_note}（基底 {base_note}，被替换的版本已快照至 {backup}）。当前 {new_bytes} 字节 / {lines} 行。'
        elif cut_note:
            msg = f'已替换上一次写入的段落到 {rel}（{cut_note}，基底 {base_note}，旧版已快照至 {backup}）。当前 {new_bytes} 字节 / {lines} 行。'
        else:
            msg = f'已替换上一次写入的段落到 {rel}（基底 {base_note}，旧版已快照至 {backup}）。当前 {new_bytes} 字节 / {lines} 行。'
        return msg + _hanzi_receipt(content, full)

    if mode == 'replace_last_write':
        return _do_replace_last_write(None)

    if mode == 'replace_last_chapter' and opening is None and _last_append_has_backup(rel):
        return _do_replace_last_write(
            '已自动降级为 replace_last_write：正文不以章节标题开头，只替换上一次追加的半章，上半章已保留'
        )

    if (
        last is not None
        and opening is not None
        and heading_count == 1
        and opening == last[0]
        and has_prior_chapters
    ):
        return _do_replace(True, last[0])

    if mode == 'overwrite' and last is not None and opening is not None and opening == last[0] + 1:
        note = (
            f'已自动转为安全追加：检测到 mode="overwrite"，但正文是第 {opening} 章、'
            f'原文末章是第 {last[0]} 章，已改为追加第 {opening} 章'
        )
        return _do_append(note)

    if mode == 'replace_last_chapter':
        if last is not None and has_prior_chapters:
            return _do_replace(False, last[0])
        _audit_write('write_novel_file', rel, 'replace_last_chapter', old_bytes, old_bytes, 'rejected', None)
        return (
            f'拒绝替换：{rel} 现有 {len(chapters)} 个章节标题，没有可锁定的前序章节。'
            'replace_last_chapter 只替换末章；请先有至少两章，或改用 append / edit_novel_file。'
        )

    if mode != 'append':
        return _reject_overwrite()
    return _do_append(None)


def _call_tool(name: str, args: dict) -> str:
    if name == 'list_novel_files':
        sub = (args.get('folder') or '').strip()
        base = _safe_path(sub) if sub else NOVEL_ROOT
        if not base.exists():
            return f'目录不存在: {sub}'
        out = []
        for p in sorted(base.rglob('*')):
            if any(part.startswith('.') or part == 'archive' for part in p.relative_to(NOVEL_ROOT).parts):
                continue
            if p.is_file() and p.suffix.lower() in ('.md', '.txt'):
                rel = p.relative_to(NOVEL_ROOT).as_posix()
                txt = p.read_text(encoding='utf-8', errors='replace')
                out.append(f'{rel}  ({len(txt.splitlines())} 行, {p.stat().st_size:,} 字节)')
        return '\n'.join(out) if out else '未找到 .md / .txt 文件'

    if name == 'read_novel_file':
        p = _safe_path(args['path'])
        if not p.exists() or not p.is_file():
            return f'文件不存在: {args["path"]}'
        all_lines = p.read_text(encoding='utf-8', errors='replace').splitlines()
        total = len(all_lines)
        tail = args.get('tail_lines')
        if tail and int(tail) > 0:
            start = max(1, total - int(tail) + 1)
            end = total
        else:
            start = max(1, int(args.get('start_line') or 1))
            end = min(total, int(args.get('end_line') or min(total, start + 199)))
        selected = [f'{i}: {all_lines[i - 1]}' for i in range(start, end + 1)]
        header = f'[{args["path"]} | 第 {start}–{end} 行 / 共 {total} 行]\n'
        return header + '\n'.join(selected)

    if name == 'grep_reference':
        target = _safe_path(args['path'])
        query = args['query']
        ctx = max(0, min(10, int(args.get('context_lines') or 3)))
        limit = max(1, min(30, int(args.get('max_matches') or 10)))
        files = [target] if target.is_file() else sorted(target.rglob('*.md'))
        pat = re.compile(query, re.IGNORECASE)
        hits = []
        for f in files:
            rel = f.relative_to(NOVEL_ROOT).as_posix() if NOVEL_ROOT in f.parents else f.name
            flines = f.read_text(encoding='utf-8', errors='replace').splitlines()
            for idx, line in enumerate(flines):
                if pat.search(line):
                    s = max(0, idx - ctx)
                    e = min(len(flines), idx + ctx + 1)
                    snippet = '\n'.join(f'  {k + 1}: {flines[k]}' for k in range(s, e))
                    hits.append(f'=== {rel} (第 {idx + 1} 行) ===\n{snippet}')
                    if len(hits) >= limit:
                        break
            if len(hits) >= limit:
                break
        return '\n\n'.join(hits) if hits else f'未找到匹配 "{query}" 的片段'

    if name == 'edit_novel_file':
        rel = str(args['path'])
        p = _safe_path(rel)
        if rel.startswith('skill:'):
            _audit_write('edit_novel_file', rel, 'edit', 0, 0, 'rejected', None)
            raise ValueError('技能文件为只读')
        if not p.exists():
            _audit_write('edit_novel_file', rel, 'edit', 0, 0, 'missing', None)
            return f'文件不存在: {rel}'
        old_bytes = p.stat().st_size
        raw = p.read_text(encoding='utf-8')
        old_text = args['old_text']
        new_text = args['new_text']
        if old_text not in raw:
            _audit_write('edit_novel_file', rel, 'edit', old_bytes, old_bytes, 'not_found', None)
            return f'未在 {rel} 中找到完全匹配的 old_text，请先用 read_novel_file 核对原文。'
        backup = _backup_before_write(p)
        updated = raw.replace(old_text, new_text, 1)
        p.write_text(updated, encoding='utf-8', newline='')
        new_bytes = p.stat().st_size
        _audit_write('edit_novel_file', rel, 'edit', old_bytes, new_bytes, 'ok', backup)
        return f'已成功更新 {rel}（总行数: {len(updated.splitlines())}）'

    if name == 'write_novel_file':
        rel = str(args['path'])
        p = _safe_path(rel)
        mode = str(args.get('mode') or 'append').strip().lower()
        if rel.startswith('skill:'):
            _audit_write('write_novel_file', rel, mode, 0, 0, 'rejected', None)
            raise ValueError('技能文件为只读')
        content = args['content']
        if not isinstance(content, str):
            raise ValueError('content 必须是字符串')
        p.parent.mkdir(parents=True, exist_ok=True)
        existed = p.exists()
        old_bytes = p.stat().st_size if existed else 0
        if existed and (old_bytes > 200 or mode in ('append', 'replace_last_chapter', 'replace_last_write')):
            return _guarded_write(rel, p, mode, content)
        backup = _backup_before_write(p) if existed else None
        payload = content if content.endswith('\n') else content + '\n'
        p.write_text(payload, encoding='utf-8', newline='')
        new_bytes = p.stat().st_size
        _audit_write('write_novel_file', rel, mode, old_bytes, new_bytes, 'ok', backup)
        return f'已成功写入 {rel}（总行数: {len(payload.splitlines())}）' + _hanzi_receipt(content, payload)

    if name == 'undo_novel_write':
        rel = str(args['path'])
        p = _safe_path(rel)
        if rel.startswith('skill:'):
            _audit_write('undo_novel_write', rel, 'restore', 0, 0, 'rejected', None)
            raise ValueError('技能文件为只读')
        if not p.exists() or not p.is_file():
            _audit_write('undo_novel_write', rel, 'restore', 0, 0, 'missing', None)
            return f'文件不存在: {rel}'
        backups = _list_backups(p)
        current = p.read_bytes()
        old_bytes = len(current)
        if not backups:
            _audit_write('undo_novel_write', rel, 'restore', old_bytes, old_bytes, 'missing_snapshot', None)
            return f'没有可撤回的快照：{rel} 在 archive/.snapshots 中没有 .bak。当前文件未改动（{old_bytes} 字节）。'
        source = backups[-1]
        source_bytes = source.read_bytes()
        old_text = current.decode('utf-8', errors='replace')
        old_lines = len(old_text.splitlines())
        old_title = _last_chapter_title(old_text)
        source_rel = source.relative_to(NOVEL_ROOT).as_posix()
        if current == source_bytes:
            _audit_write('undo_novel_write', rel, 'restore', old_bytes, old_bytes, 'unchanged', source_rel)
            return (
                f'当前文件已与最新快照一致，无需撤回：{rel}'
                f'（{old_bytes} 字节 / {old_lines} 行，末章「{old_title}」）。'
            )
        saved = _backup_before_write(p)
        p.write_bytes(source_bytes)
        new_text = source_bytes.decode('utf-8', errors='replace')
        new_lines = len(new_text.splitlines())
        new_title = _last_chapter_title(new_text)
        new_bytes = p.stat().st_size
        _audit_write('undo_novel_write', rel, 'restore', old_bytes, new_bytes, 'ok', saved)
        return (
            f'已撤回 {rel} 到上一版（来源快照 {source_rel}）。\n'
            f'恢复前 {old_bytes} 字节 / {old_lines} 行，末章「{old_title}」\n'
            f'恢复后 {new_bytes} 字节 / {new_lines} 行，末章「{new_title}」\n'
            f'被覆盖的当前版本已另存快照 {saved or "（无）"}，可再次撤回。'
        )

    raise ValueError(f'未知工具: {name}')


class MCPHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def _send_cors(self):
        origin = self.headers.get('Origin') or '*'
        self.send_header('Access-Control-Allow-Origin', origin)
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
        req_hdrs = self.headers.get('Access-Control-Request-Headers')
        self.send_header(
            'Access-Control-Allow-Headers',
            req_hdrs if req_hdrs else 'Content-Type, Accept, Authorization, MCP-Protocol-Version, Mcp-Session-Id',
        )
        self.send_header('Access-Control-Expose-Headers', 'Mcp-Session-Id, MCP-Protocol-Version')
        self.send_header('Access-Control-Allow-Private-Network', 'true')
        self.send_header('Mcp-Session-Id', SESSION_ID)

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors()
        self.end_headers()

    def do_DELETE(self):
        self.send_response(200)
        self._send_cors()
        self.send_header('Content-Length', '0')
        self.end_headers()

    def do_GET(self):
        if self.path.startswith('/health'):
            body = json.dumps({'ok': True, 'root': str(NOVEL_ROOT)}, ensure_ascii=False).encode('utf-8')
            self.send_response(200)
            self._send_cors()
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(405)
        self._send_cors()
        self.send_header('Content-Length', '0')
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get('Content-Length') or 0)
        raw = self.rfile.read(length).decode('utf-8', errors='replace') if length else '{}'
        try:
            msg = json.loads(raw)
        except Exception as e:
            self._reply_json(400, {'jsonrpc': '2.0', 'id': None, 'error': {'code': -32700, 'message': str(e)}})
            return

        if isinstance(msg, list):
            responses = [r for m in msg if (r := self._handle_rpc(m)) is not None]
            if not responses:
                self.send_response(202)
                self._send_cors()
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            self._reply_json(200, responses)
            return

        resp = self._handle_rpc(msg)
        if resp is None:
            self.send_response(202)
            self._send_cors()
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        self._reply_json(200, resp)

    def _handle_rpc(self, msg: dict) -> dict | None:
        method = msg.get('method', '')
        req_id = msg.get('id')
        params = msg.get('params') or {}

        if req_id is None or method.startswith('notifications/'):
            return None

        if method == 'initialize':
            proto = params.get('protocolVersion') or '2025-03-26'
            return {
                'jsonrpc': '2.0',
                'id': req_id,
                'result': {
                    'protocolVersion': proto,
                    'capabilities': {'tools': {'listChanged': False}},
                    'serverInfo': {'name': 'local_novel_fs', 'version': '1.0.0'},
                    'instructions': '本机小说目录 (C:/Users/zym/Documents/Colab创作) 读写与原作检索工具。',
                },
            }

        if method == 'ping':
            return {'jsonrpc': '2.0', 'id': req_id, 'result': {}}

        if method == 'tools/list':
            return {'jsonrpc': '2.0', 'id': req_id, 'result': {'tools': TOOLS}}

        if method == 'resources/list':
            return {'jsonrpc': '2.0', 'id': req_id, 'result': {'resources': []}}

        if method == 'prompts/list':
            return {'jsonrpc': '2.0', 'id': req_id, 'result': {'prompts': []}}

        if method == 'tools/call':
            tname = params.get('name', '')
            targs = params.get('arguments') or {}
            try:
                text = _call_tool(tname, targs)
                return {
                    'jsonrpc': '2.0',
                    'id': req_id,
                    'result': {'content': [{'type': 'text', 'text': text}], 'isError': False},
                }
            except Exception as err:
                return {
                    'jsonrpc': '2.0',
                    'id': req_id,
                    'result': {'content': [{'type': 'text', 'text': f'Error: {err}'}], 'isError': True},
                }

        return {
            'jsonrpc': '2.0',
            'id': req_id,
            'error': {'code': -32601, 'message': f'Method not found: {method}'},
        }

    def _reply_json(self, status: int, payload: dict | list):
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self._send_cors()
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    NOVEL_ROOT.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer((HOST, PORT), MCPHandler)
    print(f'local_novel_mcp listening on http://{HOST}:{PORT}/mcp (root={NOVEL_ROOT})', flush=True)
    srv.serve_forever()


if __name__ == '__main__':
    sys.exit(main())
