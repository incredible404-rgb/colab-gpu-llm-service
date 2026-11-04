"""Colab CU meter: live balance, ghost-session kill switch and two-track CU bill (zero OAuth).

    python scripts/colab_meter.py status [--json]
    python scripts/colab_meter.py start [-s NAME] [-g GPU]
    python scripts/colab_meter.py stop  [-s NAME] [--all]

All account data comes from Colab's /tun/m endpoints through colab_sync (a signed-in Colab tab in Edge, via
Playwriter or playwright-cli). Meter state lives in .colab-meter-state.json next to this repo's README.

Two-track bill printed by `stop`:
  track 1  cloud ledger   max(0, start_balance - end_balance)         what Colab really deducted
  track 2  wall clock     duration_hours * rate_hourly                 what the burn rate predicts
The ledger can lag a few minutes behind the real burn, so the two tracks are shown side by side.
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import colab_sync  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parents[1]
STATE_FILE = ROOT_DIR / '.colab-meter-state.json'


def _load_state():
    if not STATE_FILE.exists():
        return {'sessions': []}
    try:
        data = json.loads(STATE_FILE.read_text(encoding='utf-8'))
    except ValueError:
        return {'sessions': []}
    if isinstance(data, dict) and isinstance(data.get('sessions'), list):
        return data
    return {'sessions': []}


def _save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='')


def start_meter(label, gpu=None):
    """Record the opening balance and time for a session (replaces an earlier record with the same label)."""
    snap = colab_sync.colab_status()
    record = {
        'label': label,
        'gpu': gpu or (snap['assignments'][0].get('variant') if snap['assignments'] else None),
        'start_time': time.time(),
        'start_time_iso': datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds'),
        'start_balance': snap['currentBalance'],
        'rate_hourly': snap['consumptionRateHourly'],
        'endpoints': [a['endpoint'] for a in snap['assignments'] if a.get('endpoint')],
    }
    state = _load_state()
    state['sessions'] = [s for s in state['sessions'] if s.get('label') != label] + [record]
    _save_state(state)
    print('Meter started: %s | GPU: %s | Start balance: %.2f CU | Burn rate: %.2f CU/h | %s' % (
        label, record['gpu'] or '-', record['start_balance'], record['rate_hourly'], record['start_time_iso']))
    return record


def _bill(session, end_balance, live_rate, now):
    """Two-track bill lines for one metered session."""
    dur_sec = max(0.0, now - session['start_time'])
    rate = live_rate if live_rate and live_rate > 0 else (session.get('rate_hourly') or 0.0)
    ledger = max(0.0, session['start_balance'] - end_balance)
    wall = dur_sec / 3600 * rate
    h, rem = divmod(int(dur_sec), 3600)
    return [
        'CU bill for "%s" (%s) - %dh%02dm%02ds' % (session['label'], session.get('gpu') or '-', h, rem // 60, rem % 60),
        '  track 1 (cloud ledger): %.2f -> %.2f CU = %.2f CU' % (session['start_balance'], end_balance, ledger),
        '  track 2 (wall clock):   %.0f s / 3600 x %.2f CU/h = %.2f CU' % (dur_sec, rate, wall),
    ]


def stop_and_report(session=None, kill_all=False):
    """Kill the target (all when kill_all or no recorded endpoints) assignments, verify 0 active / 0.00 CU/h
    and print the two-track bill. Returns 0 when clean, 1 otherwise."""
    state = _load_state()
    sessions = state['sessions']
    chosen = [s for s in sessions if session is None or s.get('label') == session]
    if session and not chosen:
        print('No meter record named "%s"; killing by live assignments only.' % session)
    endpoints = None
    if session and not kill_all:
        endpoints = [e for s in chosen for e in s.get('endpoints', [])] or None

    result = colab_sync.colab_unassign(endpoints)
    now = time.time()
    before, after = result['before'], result['after']
    print('Before: ' + colab_sync.format_status(before).split('\n')[0])
    for item in result['released']:
        print('  unassign %s: %s (HTTP %s, step %s)' % (item['endpoint'], 'OK' if item['ok'] else 'FAILED', item['status'], item['step']))
    if after is None or 'error' in after:
        print('Verification failed: %s' % (after,), file=sys.stderr)
        return 1
    print('After:  ' + colab_sync.format_status(after).split('\n')[0])
    for a in after['assignments']:
        print('  still active: %s | %s | %s' % (a.get('endpoint'), a.get('accelerator'), a.get('machineShape')))
    clean = colab_sync.is_clean(after)
    if clean:
        print('Verified: Active assignments: 0 | Burn rate: 0.00 CU/h')
    else:
        print('NOT CLEAN: %d assignment(s) still active, burn rate %.2f CU/h' % (after['assignmentsCount'], after['consumptionRateHourly']), file=sys.stderr)

    if chosen:
        for s in chosen:
            print('\n'.join(_bill(s, after['currentBalance'], before['consumptionRateHourly'], now)))
        if clean:
            state['sessions'] = [s for s in sessions if s not in chosen]
            _save_state(state)
    else:
        print('No meter record (run `colab_meter.py start` first); balance now %.2f CU.' % after['currentBalance'])
    return 0 if clean else 1


def cmd_status(args):
    snap = colab_sync.colab_status()
    if args.json:
        snap['meter_sessions'] = _load_state()['sessions']
        print(json.dumps(snap, ensure_ascii=False, indent=2))
        return 0
    print(colab_sync.format_status(snap))
    for s in _load_state()['sessions']:
        elapsed = (time.time() - s['start_time']) / 3600
        print('  meter: %s | GPU %s | started %s | %.2f CU at start | %.2f h elapsed' % (
            s['label'], s.get('gpu') or '-', s.get('start_time_iso'), s['start_balance'], elapsed))
    return 0


def cmd_start(args):
    start_meter(args.session or 'default', args.gpu)
    return 0


def cmd_stop(args):
    return stop_and_report(session=args.session, kill_all=args.all)


def main(argv=None):
    ap = argparse.ArgumentParser(description='Colab CU meter (zero OAuth, /tun/m)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('status', help='balance, burn rate and active assignments')
    p.add_argument('--json', action='store_true')
    p.set_defaults(fn=cmd_status)
    p = sub.add_parser('start', help='record the opening balance and time')
    p.add_argument('-s', '--session')
    p.add_argument('-g', '--gpu')
    p.set_defaults(fn=cmd_start)
    p = sub.add_parser('stop', help='kill assignments, verify 0 active / 0.00 CU/h, print the two-track bill')
    p.add_argument('-s', '--session')
    p.add_argument('--all', action='store_true', help='kill every active assignment')
    p.set_defaults(fn=cmd_stop)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == '__main__':
    sys.exit(main())
