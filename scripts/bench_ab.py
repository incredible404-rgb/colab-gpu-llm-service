"""Repeatable llama-server benchmark + functional gates for the Colab service.

Stdlib only. Measures the already-running server; it never starts or stops Colab.

  python scripts/bench_ab.py decode  --label mtp3          # 24 prompts, greedy, 256 tokens each
  python scripts/bench_ab.py recall  --label mtp3          # needle at 8k/32k/128k/216k tokens
  python scripts/bench_ab.py tools   --label mtp3          # 48 repeated tool calls
  python scripts/bench_ab.py all     --label mtp3          # the three above
  python scripts/bench_ab.py compare a.json b.json         # speed delta + greedy-output agreement

Method (borrowed from a dual-GPU SGLang tuning write-up, adapted to llama.cpp):
- Greedy (temperature 0), forced 256 output tokens, cache_prompt off: every run does the same work.
- Run-to-run speed varies a lot, so judge A/B by alternating windows (A, B, A, B) and by
  the spread inside one label, not by one run.
- Quality: run the SAME config twice and `compare` them first. That agreement is the noise
  floor; a change only counts when it falls clearly below it.
- Gates before a candidate is accepted: recall passes at every length, tool calls 48/48.

Base URL: --base-url, else BENCH_BASE_URL, else http://$TAILNET_FQDN:443/v1.
API key (if the server has one): LLAMA_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DECODE_TOKENS = 256
RECALL_LENGTHS = (8_000, 32_000, 128_000, 216_000)
TOOL_ROUNDS = 48

DECODE_PROMPTS = [
    '用 Python 写一个带 LRU 淘汰的缓存类，并给出用法示例。',
    '解释 TCP 三次握手和四次挥手，说明为什么挥手要四次。',
    '写一段 300 字的江南雨天散文。',
    '证明根号 2 是无理数。',
    'Write a Rust function that merges two sorted vectors without extra allocation beyond the output.',
    '比较 B 树和 LSM 树在写放大上的差异。',
    '给一个 12 岁孩子讲讲什么是黑洞。',
    '写一个 SQL：每个部门薪资排名前三的员工。',
    '用 TypeScript 实现一个简单的发布订阅事件总线。',
    '说明 Transformer 里 KV cache 为什么能加速解码。',
    '续写小说开头：深夜，灯塔的灯忽然灭了。',
    '求解 x^3 - 6x^2 + 11x - 6 = 0 并写出步骤。',
    'Explain the difference between a mutex and a semaphore with a short example.',
    '写一份 Linux 排查磁盘占满的步骤清单。',
    '用 Go 写一个并发限流的 worker pool。',
    '概括一下快速排序的最坏情况和改进办法。',
    '写一首七言绝句，主题是秋夜读书。',
    '解释 CAP 定理，并举一个取舍 AP 的真实系统。',
    '设计一个短链接服务，列出主要组件和瓶颈。',
    '写一个正则表达式匹配中国大陆手机号，并解释每一段。',
    '解释梯度消失，以及残差连接为何有帮助。',
    'Write a bash script that rotates logs older than 7 days and keeps a count.',
    '用通俗的话讲讲量化 GGUF 的 Q4_K 和 Q8_0 有什么不同。',
    '写一个 Python 装饰器，给函数加重试和指数退避。',
]

WORDS = ('amber basin cobalt dune ember fjord glacier harbor iris jasper kelp lantern meadow '
         'nectar orchid prism quartz ridge summit tundra umber valley willow yarrow zephyr').split()


def default_base_url() -> str:
    url = os.environ.get('BENCH_BASE_URL', '').strip()
    if url:
        return url
    fqdn = os.environ.get('TAILNET_FQDN', '').strip()
    if not fqdn:
        sys.exit('set --base-url, BENCH_BASE_URL or TAILNET_FQDN')
    return 'http://%s:443/v1' % fqdn


class Client:
    def __init__(self, base_url: str, timeout: float = 900):
        self.base = base_url.rstrip('/')
        self.root = self.base[:-3] if self.base.endswith('/v1') else self.base
        self.timeout = timeout
        key = os.environ.get('LLAMA_API_KEY', '').strip()
        self.headers = {'Content-Type': 'application/json'}
        if key:
            self.headers['Authorization'] = 'Bearer ' + key

    def _call(self, url: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, headers=self.headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError('HTTP %s from %s: %s' % (e.code, url, e.read()[:300].decode('utf-8', 'replace')))

    def model(self) -> str:
        return self._call(self.base + '/models')['data'][0]['id']

    def chat(self, body: dict) -> dict:
        return self._call(self.base + '/chat/completions', body)

    def tokenize(self, text: str) -> int:
        return len(self._call(self.root + '/tokenize', {'content': text})['tokens'])


def timing_row(resp: dict) -> dict:
    tm = resp.get('timings') or {}
    usage = resp.get('usage') or {}
    return {
        'tps': tm.get('predicted_per_second'),
        'prefill_tps': tm.get('prompt_per_second'),
        'completion_tokens': usage.get('completion_tokens'),
        'draft_n': tm.get('draft_n'),
        'draft_accepted': tm.get('draft_n_accepted'),
    }


def run_decode(c: Client, model: str, effort: str) -> dict:
    items = []
    for i, prompt in enumerate(DECODE_PROMPTS):
        resp = c.chat({
            'model': model, 'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0, 'top_k': 1, 'max_tokens': DECODE_TOKENS, 'ignore_eos': True,
            'cache_prompt': False, 'reasoning_effort': effort, 'seed': 1,
        })
        msg = resp['choices'][0]['message']
        row = timing_row(resp)
        row['text'] = (msg.get('reasoning_content') or '') + '\x00' + (msg.get('content') or '')
        items.append(row)
        print('decode %2d/%d  %6.1f tok/s  tokens=%s' % (i + 1, len(DECODE_PROMPTS), row['tps'] or 0, row['completion_tokens']))
    return summarize_decode(items)


def summarize_decode(items: list[dict]) -> dict:
    speeds = [r['tps'] for r in items if r['tps']]
    drafted = sum(r['draft_n'] or 0 for r in items)
    accepted = sum(r['draft_accepted'] or 0 for r in items)
    short = [i for i, r in enumerate(items) if (r['completion_tokens'] or 0) < DECODE_TOKENS]
    return {
        'median_tps': statistics.median(speeds) if speeds else None,
        'mean_tps': statistics.fmean(speeds) if speeds else None,
        'min_tps': min(speeds) if speeds else None,
        'max_tps': max(speeds) if speeds else None,
        'draft_acceptance': (accepted / drafted) if drafted else None,
        'stopped_early': short,  # ignore_eos not honored for these prompts; their speed is not comparable
        'items': items,
    }


def build_filler(chars: int, rng: random.Random) -> list[str]:
    paras, total = [], 0
    while total < chars:
        sents = []
        for _ in range(6):
            w = [rng.choice(WORDS) for _ in range(rng.randint(8, 14))]
            sents.append('The %s %s recorded %d units near the %s on day %d.'
                         % (w[0], w[1], rng.randint(10, 9999), ' '.join(w[2:6]), rng.randint(1, 365)))
        para = ' '.join(sents)
        paras.append(para)
        total += len(para) + 2
    return paras


def run_recall(c: Client, model: str, effort: str, lengths) -> list[dict]:
    rng = random.Random(20261109)
    sample = '\n\n'.join(build_filler(3000, rng))
    cpt = len(sample) / max(1, c.tokenize(sample))
    results = []
    for target in lengths:
        code = '%s-%d' % (rng.choice(WORDS).upper(), rng.randint(100000, 999999))
        paras = build_filler(int(target * cpt), rng)
        paras.insert(len(paras) // 2, 'IMPORTANT: the vault passphrase is %s. Remember it exactly.' % code)
        prompt = '\n\n'.join(paras) + '\n\n上文里提到的 vault passphrase 是什么？只回答口令本身。'
        actual = c.tokenize(prompt)
        t0 = time.monotonic()
        resp = c.chat({
            'model': model, 'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0, 'max_tokens': 2048, 'cache_prompt': False, 'reasoning_effort': effort,
        })
        answer = resp['choices'][0]['message'].get('content') or ''
        row = {'target_tokens': target, 'prompt_tokens': actual, 'seconds': round(time.monotonic() - t0, 1),
               'passed': code in answer, 'finish': resp['choices'][0].get('finish_reason'),
               'answer_head': answer[:80], **timing_row(resp)}
        results.append(row)
        print('recall %7d tok  %s  %.0fs  prefill %s tok/s'
              % (actual, 'PASS' if row['passed'] else 'FAIL', row['seconds'], row['prefill_tps']))
    return results


TOOL_SPEC = [{'type': 'function', 'function': {
    'name': 'get_weather', 'description': 'Get the current weather for a city.',
    'parameters': {'type': 'object', 'properties': {
        'city': {'type': 'string'}, 'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']}},
        'required': ['city', 'unit']}}}]
CITIES = ['Hangzhou', 'Reykjavik', 'Lima', 'Nairobi', 'Osaka', 'Tromso', 'Quito', 'Perth', 'Oslo', 'Cusco']


def check_tool_call(resp: dict, city: str) -> str | None:
    """Return None when the call is correct, else a short failure reason."""
    calls = resp['choices'][0]['message'].get('tool_calls') or []
    if not calls:
        return 'no_tool_call'
    fn = calls[0].get('function') or {}
    if fn.get('name') != 'get_weather':
        return 'wrong_name'
    try:
        args = json.loads(fn.get('arguments') or '')
    except json.JSONDecodeError:
        return 'bad_json'
    if str(args.get('city', '')).lower() != city.lower():
        return 'wrong_city'
    if args.get('unit') != 'celsius':
        return 'wrong_unit'
    return None


def run_tools(c: Client, model: str, effort: str, rounds: int) -> dict:
    rng = random.Random(7)
    failures: dict[str, int] = {}
    for i in range(rounds):
        city = rng.choice(CITIES)
        resp = c.chat({
            'model': model, 'tools': TOOL_SPEC, 'temperature': 0, 'max_tokens': 1024,
            'cache_prompt': False, 'reasoning_effort': effort,
            'messages': [{'role': 'user', 'content': '查一下 %s 现在的天气，用摄氏度。' % city}],
        })
        bad = check_tool_call(resp, city)
        if bad:
            failures[bad] = failures.get(bad, 0) + 1
        print('tools %2d/%d  %s' % (i + 1, rounds, bad or 'ok'))
    return {'rounds': rounds, 'failed': sum(failures.values()), 'failures': failures}


def common_prefix_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n / max(len(a), len(b))


def compare(a: dict, b: dict) -> str:
    da, db = a.get('decode'), b.get('decode')
    lines = ['A=%s  B=%s' % (a.get('label'), b.get('label'))]
    if da and db:
        ma, mb = da['median_tps'], db['median_tps']
        lines.append('decode median tok/s: A %.1f  B %.1f  (%+.1f%%)' % (ma, mb, (mb / ma - 1) * 100))
        lines.append('decode range: A %.1f-%.1f  B %.1f-%.1f'
                     % (da['min_tps'], da['max_tps'], db['min_tps'], db['max_tps']))
        if da['draft_acceptance'] is not None and db['draft_acceptance'] is not None:
            lines.append('MTP acceptance: A %.1f%%  B %.1f%%' % (da['draft_acceptance'] * 100, db['draft_acceptance'] * 100))
        ratios = [common_prefix_ratio(x['text'], y['text']) for x, y in zip(da['items'], db['items'])]
        lines.append('greedy output agreement (common-prefix ratio): mean %.3f  identical %d/%d'
                     % (statistics.fmean(ratios), sum(r == 1.0 for r in ratios), len(ratios)))
        lines.append('  (run the same config twice and compare first: that is the noise floor)')
    for name, r in (('A', a), ('B', b)):
        if r.get('recall'):
            lines.append('recall %s: %s' % (name, ' '.join(
                '%dk:%s' % (x['prompt_tokens'] // 1000, 'ok' if x['passed'] else 'FAIL') for x in r['recall'])))
        if r.get('tools'):
            t = r['tools']
            lines.append('tools  %s: %d/%d failed %s' % (name, t['failed'], t['rounds'], t['failures']))
    return '\n'.join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cmd', choices=['decode', 'recall', 'tools', 'all', 'compare'])
    p.add_argument('files', nargs='*', help='compare: two result json files')
    p.add_argument('--label', default='run')
    p.add_argument('--base-url')
    p.add_argument('--effort', default='low', help='reasoning_effort sent with each request (default low)')
    p.add_argument('--lengths', default=','.join(map(str, RECALL_LENGTHS)), help='recall prompt sizes in tokens')
    p.add_argument('--tool-rounds', type=int, default=TOOL_ROUNDS)
    p.add_argument('--out-dir', default=str(Path(__file__).resolve().parent.parent / 'debug-runs'))
    args = p.parse_args(argv)

    if args.cmd == 'compare':
        if len(args.files) != 2:
            p.error('compare needs two result files')
        print(compare(*[json.loads(Path(f).read_text(encoding='utf-8')) for f in args.files]))
        return 0

    c = Client(args.base_url or default_base_url())
    model = c.model()
    print('model:', model)
    result = {'label': args.label, 'model': model, 'effort': args.effort,
              'when': time.strftime('%Y-%m-%d %H:%M:%S %z')}
    if args.cmd in ('decode', 'all'):
        d = result['decode'] = run_decode(c, model, args.effort)
        acc = '%.1f%%' % (d['draft_acceptance'] * 100) if d['draft_acceptance'] is not None else 'n/a'
        print('decode median %.1f tok/s (range %.1f-%.1f)  MTP acceptance %s  stopped early: %s'
              % (d['median_tps'], d['min_tps'], d['max_tps'], acc, d['stopped_early']))
    if args.cmd in ('recall', 'all'):
        result['recall'] = run_recall(c, model, args.effort, [int(x) for x in args.lengths.split(',')])
    if args.cmd in ('tools', 'all'):
        t = result['tools'] = run_tools(c, model, args.effort, args.tool_rounds)
        print('tools failed %d/%d %s' % (t['failed'], t['rounds'], t['failures']))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / ('bench-%s-%s.json' % (args.label, time.strftime('%Y%m%d-%H%M%S')))
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    print('saved', path)
    failed = (result.get('tools') or {}).get('failed', 0) or any(not r['passed'] for r in result.get('recall', []))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
