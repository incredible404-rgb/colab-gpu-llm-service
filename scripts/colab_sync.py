"""Push generated notebook cells into a Colab tab of the signed-in Edge (Playwriter, falling back to playwright-cli -s=edge).

Shared by make_*_notebook.py --sync-colab. The Colab tab is chosen by Drive file id: a tab on any other
notebook is never touched. Cells are matched by type, surplus cells are deleted, texts are set and read back,
the runtime type (GPU model / machine shape) is set in the notebook metadata, and the notebook is saved with
saveNotebook(). Without --run-cell1 the tab is then reloaded from Drive and every cell is hashed again, so
"saved" means Drive really holds the new cells.

Also wraps Colab's /tun/m endpoints (zero OAuth, runs inside a signed-in colab.research.google.com tab):
colab_status() reads balance / burn rate / assignments, colab_unassign() kills assignments with the two-step
GET token + POST handshake and re-checks until 0 active / 0.00 CU/h. Those calls use drive id "__ANY__": any
open Colab tab is reused untouched, otherwise a temporary tab is opened and closed again.
Standalone: python scripts/colab_sync.py --status | --release-all
"""
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import zlib
from pathlib import Path

CLI_JS = Path(os.environ.get('APPDATA', str(Path.home() / 'AppData/Roaming'))) / 'npm/node_modules/@playwright/cli/playwright-cli.js'
CLI_TIMEOUT = 120  # the bridge hangs forever when it drops, so every call needs an upper bound
PW_HELPER = Path(__file__).with_name('colab_pw.mjs')
_target = {}
_backend = None  # 'playwriter' until it fails once in this process, then 'bridge'
ANY_TAB = '__ANY__'
ANY_URL = 'https://colab.research.google.com/'
COLAB_TAB_RE = re.compile(r'^- (\d+): (\(current\) )?.*https://colab\.research\.google\.com(/|\s|$)', re.M)

SYNC_JS = r"""async () => {
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
if (!location.href.includes(__DRIVE_ID__)) return { error: 'wrong tab', href: location.href };
for (let i = 0; i < 80 && !window.colab?.global?.notebook; i++) await sleep(500);
const nb = window.colab?.global?.notebook;
if (!nb) return { error: 'notebook not ready', href: location.href };
const model = nb.getNotebookModel();
const bytes = Uint8Array.from(atob("__PAYLOAD__"), c => c.charCodeAt(0));
const ds = new DecompressionStream("deflate");
const writer = ds.writable.getWriter();
writer.write(bytes);
writer.close();
const want = JSON.parse(await new Response(ds.readable).text());
let cells = nb.cells;
const keep = [];
let scan = 0;
for (const w of want) {
  while (scan < cells.length && cells[scan].getType() !== w.type) scan++;
  if (scan >= cells.length) return { error: 'cell types cannot be matched', want: want.map(w => w.type), have: cells.map(c => c.getType()) };
  keep.push(cells[scan]);
  scan++;
}
const drop = cells.filter(c => !keep.includes(c));
if (drop.length) {
  nb.removeCells(drop);
  for (let i = 0; i < 20 && nb.cells.length !== want.length; i++) await sleep(250);
}
cells = nb.cells;
if (cells.length !== want.length) return { error: 'cell count mismatch after reshape', have: cells.length, want: want.length };
for (let i = 0; i < want.length; i++) cells[i].setText(want[i].text);
const textsMatch = want.map((w, i) => cells[i].getText() === w.text);
const rt = __RUNTIME__;
let runtimeChanged = false;
if (rt && (model.getAcceleratorModel() !== rt.acceleratorModel || model.getMachineShape() !== rt.machineShape || model.getAccelerator() !== rt.accelerator)) {
  model.setRuntime({ accelerator: rt.accelerator, acceleratorModel: rt.acceleratorModel, machineShape: rt.machineShape });
  runtimeChanged = true;
}
window.onbeforeunload = null;
let saveError = null;
try { await Promise.race([nb.saveNotebook(), sleep(20000)]); } catch (e) { saveError = String(e); }
for (let i = 0; i < 40 && model.isBackingFileDirty(); i++) await sleep(500);
let released = false;
let ran = false;
if (__RUN_CELL1__) {
  const runBtn = document.querySelectorAll("colab-run-button")[0];
  const inner = runBtn?.shadowRoot?.querySelector("button") || runBtn?.querySelector("button");
  if (inner) { inner.click(); ran = true; }
} else if (__RELEASE__ && nb.kernel?.isConnected?.()) {
  try { await Promise.race([nb.kernel.unassignCurrentVm({ skipConfirm: true }), sleep(8000)]); released = true; } catch (e) {}
}
return {
  href: location.href,
  updated: textsMatch.filter(Boolean).length,
  total: want.length,
  removedCells: drop.length,
  types: cells.map(c => c.getType()),
  acceleratorModel: model.getAcceleratorModel(),
  machineShape: model.getMachineShape(),
  accelerator: model.getAccelerator(),
  runtimeChanged,
  saveError,
  dirtyAfterSave: model.isBackingFileDirty(),
  kernelConnected: !!nb.kernel?.isConnected?.(),
  ran,
  released
};
}"""

READBACK_JS = r"""async () => {
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
if (!location.href.includes(__DRIVE_ID__)) return { error: 'wrong tab', href: location.href };
for (let i = 0; i < 120 && !window.colab?.global?.notebook?.cells?.length; i++) await sleep(500);
const nb = window.colab?.global?.notebook;
if (!nb) return { error: 'notebook not ready', href: location.href };
const model = nb.getNotebookModel();
const hex = (buf) => [...new Uint8Array(buf)].map(b => b.toString(16).padStart(2, '0')).join('');
const hashes = [];
for (const c of nb.cells) hashes.push(hex(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(c.getText()))));
return { hashes, types: nb.cells.map(c => c.getType()), acceleratorModel: model.getAcceleratorModel(), machineShape: model.getMachineShape(), kernelConnected: !!nb.kernel?.isConnected?.() };
}"""

RELOAD_JS = "() => { window.onbeforeunload = null; setTimeout(() => location.reload(), 200); return 'reloading'; }"


def _cli(*args, timeout=CLI_TIMEOUT):
    return subprocess.run(['node', str(CLI_JS), '-s=edge', *args], capture_output=True, text=True,
                          encoding='utf-8', errors='replace', timeout=timeout)


def _result_json(proc):
    """playwright-cli prints '### Result\n<json>\n### Ran Playwright code ...'; return the parsed json part."""
    head = proc.stdout.split('### Ran Playwright code')[0]
    body = head.split('### Result', 1)[-1].strip()
    if not body or '### Error' in head:
        raise RuntimeError('playwright-cli eval failed:\n' + head.strip() + ('\nSTDERR: ' + proc.stderr if proc.stderr else ''))
    return json.loads(body)


class _PlaywriterUnavailable(RuntimeError):
    pass


def _pw_eval(js, timeout=CLI_TIMEOUT, target=None):
    target = target or _target
    try:
        proc = subprocess.run(['node', str(PW_HELPER), target['drive_id'], target['url']], input=js,
                              capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout)
    except subprocess.TimeoutExpired:
        raise _PlaywriterUnavailable('no answer within %ss' % timeout)
    lines = [l for l in proc.stdout.splitlines() if l.startswith('PWJSON ')]
    if not lines:
        raise _PlaywriterUnavailable((proc.stdout + proc.stderr).strip()[-400:] or 'helper printed nothing')
    out = json.loads(lines[-1][len('PWJSON '):])
    if 'pwError' in out:
        raise _PlaywriterUnavailable(out['pwError'])
    return out['result']


def _eval(js):
    global _backend
    if _backend != 'bridge':
        try:
            result = _pw_eval(js)
            _backend = 'playwriter'
            return result
        except _PlaywriterUnavailable as e:
            print('Playwriter unavailable (%s); falling back to playwright-cli -s=edge' % e, file=sys.stderr)
            _backend = 'bridge'
            _select_tab(_target['drive_id'], _target['url'])
    if len(js) > 30000:
        raise RuntimeError('eval payload is %d chars; the Windows command line limit (32767) is too close' % len(js))
    return _result_json(_cli('eval', js))


def _select_tab(drive_id, url):
    """Make the tab on this Drive id the current one (open it when absent)."""
    pattern = re.compile(r'^- (\d+): (\(current\) )?.*' + re.escape(drive_id), re.M)
    listing = _cli('tab-list', timeout=60).stdout
    found = pattern.search(listing)
    if found is None:
        _cli('tab-new', url, timeout=90)
        time.sleep(6)
        found = pattern.search(_cli('tab-list', timeout=60).stdout)
        if found is None:
            raise RuntimeError('could not open the Colab tab for ' + drive_id)
        return
    if found.group(2) is None:
        _cli('tab-select', found.group(1), timeout=60)


METER_JS = r"""async () => {
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const op = __OP__;
const HDR = { "Accept": "application/json", "X-Colab-Client-Agent": "colab-cli" };
const strip = (text) => text.replace(/^\)\]\}'\s*/, '');
const withTimeout = (p, ms, what) => Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error(what + ' timed out after ' + ms + ' ms')), ms))]);
const getJson = async (path) => {
  const r = await withTimeout(fetch(path, { headers: HDR, credentials: 'include' }), 15000, 'GET ' + path);
  const text = await withTimeout(r.text(), 15000, 'read ' + path);
  let data = null;
  try { data = JSON.parse(strip(text)); } catch (e) {}
  return { status: r.status, data, head: data ? '' : text.slice(0, 120) };
};
const snapshot = async () => {
  let c, a;
  try { [c, a] = await Promise.all([getJson('/tun/m/ccu-info?authuser=0'), getJson('/tun/m/assignments?authuser=0')]); }
  catch (e) { return { error: 'snapshot EXCEPTION ' + String(e) }; }
  if (!c.data) return { error: 'ccu-info HTTP ' + c.status + ' ' + c.head };
  if (!a.data) return { error: 'assignments HTTP ' + a.status + ' ' + a.head };
  const list = (a.data.assignments || []).map(x => ({ endpoint: x.endpoint, accelerator: x.accelerator, variant: x.variant, machineShape: x.machineShape }));
  return {
    currentBalance: c.data.currentBalance, consumptionRateHourly: c.data.consumptionRateHourly,
    assignmentsCount: c.data.assignmentsCount, eligibleGpus: c.data.eligibleGpus || [], eligibleTpus: c.data.eligibleTpus || [],
    assignments: list,
  };
};
const before = await snapshot();
if (before.error || op.op === 'status') return before;
const targets = op.endpoints && op.endpoints.length ? op.endpoints : before.assignments.map(x => x.endpoint).filter(Boolean);
const released = [];
for (const ep of targets) {
  try {
    const url = '/tun/m/unassign/' + encodeURIComponent(ep) + '?authuser=0';
    const g = await getJson(url);
    const token = g.data && g.data.token;
    if (!token) { released.push({ endpoint: ep, ok: false, step: 'GET', status: g.status }); continue; }
    const r = await withTimeout(fetch(url, { method: 'POST', credentials: 'include', headers: Object.assign({}, HDR, { 'X-Goog-Colab-Token': token }) }), 15000, 'POST ' + url);
    released.push({ endpoint: ep, ok: r.ok, step: 'POST', status: r.status });
  } catch (e) { released.push({ endpoint: ep, ok: false, step: 'EXCEPTION', status: String(e) }); }
}
let after = null;
for (let i = 0; i < 3; i++) {
  await sleep(1500);
  after = await snapshot();
  if (!after.error && after.assignmentsCount === 0 && after.consumptionRateHourly === 0 && after.assignments.length === 0) break;
}
return { before, released, after };
}"""


def _eval_any(js):
    """Run js in any Colab tab. An open colab.research.google.com tab is reused untouched; without one a
    temporary tab is opened and closed again right after the evaluation."""
    global _backend
    target = {'drive_id': ANY_TAB, 'url': ANY_URL}
    if _backend != 'bridge':
        try:
            result = _pw_eval(js, target=target)
            _backend = 'playwriter'
            return result
        except _PlaywriterUnavailable as e:
            print('Playwriter unavailable (%s); falling back to playwright-cli -s=edge' % e, file=sys.stderr)
            _backend = 'bridge'
    return _bridge_eval_any(js)


def _bridge_eval_any(js):
    """playwright-cli fallback: reuse an open Colab tab, else open a temporary one and tab-close it again."""
    listing = _cli('tab-list', timeout=60).stdout
    current = re.search(r'^- (\d+): \(current\) ', listing, re.M)
    found = COLAB_TAB_RE.search(listing)
    opened = None
    try:
        if found is None:
            _cli('tab-new', ANY_URL, timeout=90)
            time.sleep(6)
            found = COLAB_TAB_RE.search(_cli('tab-list', timeout=60).stdout)
            if found is None:
                raise RuntimeError('could not open a temporary Colab tab')
            opened = found.group(1)
        if found.group(2) is None:
            _cli('tab-select', found.group(1), timeout=60)
        return _result_json(_cli('eval', js))
    finally:
        if opened is not None:
            _cli('tab-close', opened, timeout=60)
        if current is not None and current.group(1) != opened:
            _cli('tab-select', current.group(1), timeout=60)


def _meter_call(op):
    out = _eval_any(METER_JS.replace('__OP__', json.dumps(op)))
    if isinstance(out, dict) and 'error' in out:
        raise RuntimeError('Colab /tun/m call failed: ' + json.dumps(out, ensure_ascii=False))
    return out


def colab_status():
    """Zero-OAuth live account read: currentBalance, consumptionRateHourly, assignmentsCount, eligibleGpus,
    eligibleTpus and assignments[{endpoint, accelerator, variant, machineShape}]."""
    return _meter_call({'op': 'status'})


def colab_unassign(endpoints=None):
    """Kill the given endpoints (all active assignments when None) via GET token + POST, then re-check
    /tun/m/ccu-info and /tun/m/assignments (1.5 s apart, up to 3 times). Returns {before, released, after}."""
    return _meter_call({'op': 'unassign', 'endpoints': list(endpoints) if endpoints else None})


def format_status(snap):
    """Single summary line plus one line per active assignment."""
    lines = ['Active assignments: %d | Burn rate: %.2f CU/h | Balance: %.2f CU | Eligible GPUs: %s' % (
        snap['assignmentsCount'], snap['consumptionRateHourly'], snap['currentBalance'],
        ', '.join(snap['eligibleGpus']) or '-')]
    for a in snap['assignments']:
        lines.append('  - %s | %s | %s | %s' % (a.get('endpoint'), a.get('accelerator'), a.get('variant'), a.get('machineShape')))
    return '\n'.join(lines)


def is_clean(snap):
    return (bool(snap) and snap.get('assignmentsCount') == 0 and snap.get('consumptionRateHourly') == 0
            and isinstance(snap.get('assignments'), list) and len(snap['assignments']) == 0)


def _want_cells(nb_cells):
    return [{'type': 'text' if c['cell_type'] == 'markdown' else 'code', 'text': ''.join(c['source'])} for c in nb_cells]


def sync_to_colab(nb, drive_id, run_cell1=False, release_runtime=False, label=None):
    """nb is the notebook dict; returns the final sync report. Raises RuntimeError on any mismatch."""
    url = 'https://colab.research.google.com/drive/' + drive_id
    want = _want_cells(nb['cells'])
    colab_meta = nb['metadata'].get('colab', {})
    runtime = {'accelerator': nb['metadata'].get('accelerator', 'GPU'),
               'acceleratorModel': colab_meta.get('gpuType'), 'machineShape': colab_meta.get('machine_shape')}
    payload = base64.b64encode(zlib.compress(json.dumps(want, ensure_ascii=False).encode('utf-8'), level=9)).decode('ascii')
    js = (SYNC_JS.replace('__DRIVE_ID__', json.dumps(drive_id)).replace('__PAYLOAD__', payload)
          .replace('__RUNTIME__', json.dumps(runtime)).replace('__RUN_CELL1__', 'true' if run_cell1 else 'false')
          .replace('__RELEASE__', 'true' if release_runtime else 'false'))
    _target.update(drive_id=drive_id, url=url)
    report = _eval(js)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if 'error' in report:
        raise RuntimeError('sync failed: ' + json.dumps(report, ensure_ascii=False))
    if report['updated'] != report['total'] or report['saveError'] or report['dirtyAfterSave']:
        raise RuntimeError('sync not clean (updated %s/%s, saveError=%s, dirty=%s)' %
                           (report['updated'], report['total'], report['saveError'], report['dirtyAfterSave']))
    if run_cell1:
        if not report['ran']:
            raise RuntimeError('Cell 1 run button not found')
        import colab_meter
        colab_meter.start_meter(label or 'colab-' + drive_id[:6], runtime['acceleratorModel'])
        return report
    if release_runtime:
        import colab_meter
        rc = colab_meter.stop_and_report(kill_all=True)
        if rc != 0:
            raise RuntimeError('runtime release verification failed (active assignments remain)')
    verify_from_drive(drive_id, url, want, runtime)
    return report


def verify_from_drive(drive_id, url, want, runtime):
    """Reload the tab (fresh load from Drive) and compare SHA-256 of every cell and the runtime metadata."""
    if _eval(RELOAD_JS) != 'reloading':
        raise RuntimeError('reload request was not accepted')
    time.sleep(10)
    read_js = READBACK_JS.replace('__DRIVE_ID__', json.dumps(drive_id))
    back = None
    for _ in range(6):
        try:
            back = _eval(read_js)
            if 'error' not in back:
                break
        except RuntimeError:
            pass
        time.sleep(5)
    if back is None or 'error' in back:
        raise RuntimeError('could not read the notebook back after reload: %s' % back)
    expected = [hashlib.sha256(w['text'].encode('utf-8')).hexdigest() for w in want]
    bad = [i for i, (a, b) in enumerate(zip(back['hashes'], expected)) if a != b]
    if len(back['hashes']) != len(expected) or bad:
        raise RuntimeError('Drive copy differs from the generated notebook (cells %s, count %d vs %d)' % (bad, len(back['hashes']), len(expected)))
    if runtime['acceleratorModel'] and (back['acceleratorModel'] != runtime['acceleratorModel'] or back['machineShape'] != runtime['machineShape']):
        raise RuntimeError('runtime type did not persist: %s / %s' % (back['acceleratorModel'], back['machineShape']))
    print('Drive read-back OK: %d cells identical (sha256), runtime %s / %s' % (len(expected), back['acceleratorModel'], back['machineShape']))


def sync_requested(nb, drive_id, label=None):
    """Entry point for the generator scripts: honour --sync-colab / --run-cell1 / --release-runtime."""
    if '--sync-colab' not in sys.argv:
        return None
    return sync_to_colab(nb, drive_id, run_cell1='--run-cell1' in sys.argv,
                         release_runtime='--release-runtime' in sys.argv, label=label)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='Colab account tools (zero OAuth, via /tun/m in a signed-in tab)')
    ap.add_argument('--status', action='store_true', help='print balance, burn rate and active assignments')
    ap.add_argument('--release-all', action='store_true', help='kill every active assignment and print the CU bill')
    args = ap.parse_args(argv)
    if args.release_all:
        import colab_meter
        return colab_meter.stop_and_report(kill_all=True)
    if args.status:
        print(format_status(colab_status()))
        return 0
    ap.print_help()
    return 2


if __name__ == '__main__':
    sys.exit(main())
