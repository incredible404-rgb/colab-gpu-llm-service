"""Colab CU meter: live balance, ghost-session kill switch and two-track CU bill (zero OAuth).

    python scripts/colab_meter.py status [--json]
    python scripts/colab_meter.py start [-s NAME] [-g GPU]
    python scripts/colab_meter.py stop  [-s NAME] [--all]

All account data comes from Colab's /tun/m endpoints through colab_sync (a signed-in Colab tab in Edge, via
Playwriter or playwright-cli). Meter state lives in .colab-meter-state.json next to this repo's README.

Two-track bill printed by `stop`:
  track 1  account ledger delta  max(0, start_balance - end_balance)  what the account balance dropped by
  track 2  wall clock            duration_hours * rate_hourly         what this session's burn rate predicts
Track 1 is an account-wide balance delta: it includes every instance running in the window and lags a few
minutes behind the real burn (it can even rise when CU are topped up or refunded). Track 2 is per session.
Both tracks are therefore shown side by side; with concurrent instances read track 2 as the per-session figure.

`--run-cell1` records a session while the VM is still being allocated (rate 0, no endpoints); `status` and
`stop` backfill the endpoints and the real burn rate as soon as the live snapshot shows the assignment.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import colab_sync  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parents[1]
STATE_FILE = ROOT_DIR / '.colab-meter-state.json'


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _clean_session(s):
    """Return a normalised copy of a stored session, or None when it is unusable."""
    if not isinstance(s, dict) or not isinstance(s.get('label'), str) or not s['label']:
        return None
    if not _is_num(s.get('start_time')) or not _is_num(s.get('start_balance')):
        return None
    out = dict(s)
    out['rate_hourly'] = s['rate_hourly'] if _is_num(s.get('rate_hourly')) else 0
    eps = s.get('endpoints')
    out['endpoints'] = [e for e in eps if isinstance(e, str) and e] if isinstance(eps, list) else []
    return out


def _load_state():
    if not STATE_FILE.exists():
        return {'sessions': []}
    try:
        data = json.loads(STATE_FILE.read_text(encoding='utf-8'))
    except ValueError:
        return {'sessions': []}
    if isinstance(data, dict) and isinstance(data.get('sessions'), list):
        sessions = [c for c in (_clean_session(s) for s in data['sessions']) if c is not None]
        return dict(data, sessions=sessions)
    return {'sessions': []}


def _save_state(state):
    """Atomic write: temp file next to the state file, then os.replace."""
    tmp = STATE_FILE.with_name(STATE_FILE.name + '.tmp')
    with open(tmp, 'w', encoding='utf-8', newline='') as fh:
        fh.write(json.dumps(state, indent=2, ensure_ascii=False) + '\n')
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, STATE_FILE)


def backfill_sessions(state, snap):
    """Fill in the late-arriving endpoints / burn rate of sessions recorded while the VM was still allocating.

    A session needs backfill when its rate_hourly is 0 or it has no endpoints. Endpoints are only bound when
    unambiguous: exactly one live assignment is not yet claimed by another session and exactly one session is
    waiting for one. The rate is the unclaimed portion of the live account burn rate.
    Returns True when the state changed (the caller saves it)."""
    assignments = [a for a in (snap or {}).get('assignments') or [] if a.get('endpoint')]
    if not assignments:
        return False
    live_rate = snap.get('consumptionRateHourly')
    sessions = state['sessions']
    claimed = {e for s in sessions for e in s['endpoints']}
    unclaimed = [a['endpoint'] for a in assignments if a['endpoint'] not in claimed]
    waiting = [s for s in sessions if not s['endpoints']]
    changed = False
    if len(waiting) == 1 and len(unclaimed) == 1:
        waiting[0]['endpoints'] = [unclaimed[0]]
        changed = True
    known_rate = sum(s['rate_hourly'] for s in sessions if _is_num(s.get('rate_hourly')) and s['rate_hourly'] > 0)
    need_rate = [s for s in sessions if s['endpoints'] and not s['rate_hourly'] > 0]
    if need_rate and _is_num(live_rate) and live_rate > known_rate:
        share = (live_rate - known_rate) / len(need_rate)
        for s in need_rate:
            s['rate_hourly'] = share
            changed = True
    return changed


def start_meter(label, gpu=None):
    """Record the opening balance and time for a session (replaces an earlier record with the same label)."""
    state = _load_state()
    others = [s for s in state['sessions'] if s.get('label') != label]
    claimed = {e for s in others for e in s['endpoints']}
    claimed_rate = sum(s['rate_hourly'] for s in others if _is_num(s.get('rate_hourly')) and s['rate_hourly'] > 0)
    snap = colab_sync.colab_status()
    unclaimed = [a for a in snap['assignments'] if a.get('endpoint') and a['endpoint'] not in claimed]
    live_rate = snap['consumptionRateHourly'] if _is_num(snap.get('consumptionRateHourly')) else 0.0
    own_rate = max(0.0, live_rate - claimed_rate) if unclaimed or not others else 0.0
    record = {
        'label': label,
        'gpu': gpu or (unclaimed[0].get('variant') if unclaimed else None),
        'start_time': time.time(),
        'start_time_iso': datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds'),
        'start_balance': snap['currentBalance'],
        'rate_hourly': own_rate,
        'endpoints': [a['endpoint'] for a in unclaimed],
    }
    state['sessions'] = others + [record]
    _save_state(state)
    print('Meter started: %s | GPU: %s | Start balance: %.2f CU | Burn rate: %.2f CU/h | %s' % (
        label, record['gpu'] or '-', record['start_balance'], record['rate_hourly'], record['start_time_iso']))
    return record


def _bill(session, end_balance, live_rate, now, concurrent=False):
    """Two-track bill lines for one metered session."""
    dur_sec = max(0.0, now - session['start_time'])
    own_rate = session.get('rate_hourly') or 0.0
    rate = own_rate if own_rate > 0 else (live_rate if live_rate and live_rate > 0 else 0.0)
    delta = session['start_balance'] - end_balance
    ledger = max(0.0, delta)
    wall = dur_sec / 3600 * rate
    h, rem = divmod(int(dur_sec), 3600)
    lines = [
        'CU bill for "%s" (%s) - %dh%02dm%02ds' % (session['label'], session.get('gpu') or '-', h, rem // 60, rem % 60),
        '  track 1 (account ledger delta): %.2f -> %.2f CU = %.2f CU' % (session['start_balance'], end_balance, ledger),
        '  track 2 (wall clock):           %.0f s / 3600 x %.2f CU/h = %.2f CU' % (dur_sec, rate, wall),
    ]
    if delta < 0:
        lines.append('  note: the balance ROSE by %.2f CU since the start (top-up/refund); track 1 is clamped to 0, '
                     'trust track 2.' % -delta)
    if concurrent:
        lines.append('  note: other instances ran or are metered in the same window; track 1 is an account-wide balance '
                     'delta shared by all of them, track 2 is this session only.')
    return lines


def stop_and_report(session=None, kill_all=False):
    """Release assignments, verify 0 active / 0.00 CU/h and print the two-track bill. Returns 0 when clean, 1 otherwise.

    With a named session (and not kill_all) only that session's recorded endpoints are released; an unknown
    session or one without endpoints is refused (return 1) instead of falling back to "kill everything".
    Without a name, everything is released only for kill_all or when at most one session is recorded."""
    state = _load_state()
    sessions = state['sessions']
    named = bool(session) and not kill_all
    if not session and not kill_all and len(sessions) > 1:
        print('%d meter sessions are recorded (%s); name one with -s NAME or use --all to release every active '
              'instance.' % (len(sessions), ', '.join(s['label'] for s in sessions)), file=sys.stderr)
        return 1
    chosen = [s for s in sessions if not session or s['label'] == session]
    if named and not chosen:
        print('No meter record named "%s"; nothing released. Use --all to force-release every active instance.' % session,
              file=sys.stderr)
        return 1

    endpoints = None
    if named:
        # read-only snapshot first: a session recorded during VM allocation may only now have its endpoints
        if backfill_sessions(state, colab_sync.colab_status()):
            _save_state(state)
            chosen = [s for s in state['sessions'] if s['label'] == session]
        endpoints = [e for s in chosen for e in s['endpoints']]
        if not endpoints:
            print('Meter session "%s" has no bound endpoints (VM not allocated yet or already gone); nothing released. '
                  'Use --all to force-release every active instance.' % session, file=sys.stderr)
            return 1

    result = colab_sync.colab_unassign(endpoints)
    now = time.time()
    before, after = result['before'], result['after']
    if not named and isinstance(before, dict) and backfill_sessions(state, before):
        _save_state(state)  # chosen holds the same dicts, so the bill below sees the backfilled rate
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
        sessions = state['sessions']
        concurrent = len(sessions) > 1 or len(before.get('assignments') or []) > len(chosen)
        for s in chosen:
            print('\n'.join(_bill(s, after['currentBalance'], before['consumptionRateHourly'], now, concurrent)))
        if clean:
            state['sessions'] = [s for s in sessions if s not in chosen]
            _save_state(state)
    else:
        print('No meter record (run `colab_meter.py start` first); balance now %.2f CU.' % after['currentBalance'])
    return 0 if clean else 1


def cmd_status(args):
    snap = colab_sync.colab_status()
    state = _load_state()
    if backfill_sessions(state, snap):
        _save_state(state)
    if args.json:
        snap['meter_sessions'] = state['sessions']
        print(json.dumps(snap, ensure_ascii=False, indent=2))
        return 0
    print(colab_sync.format_status(snap))
    for s in state['sessions']:
        elapsed = (time.time() - s['start_time']) / 3600
        print('  meter: %s | GPU %s | started %s | %.2f CU at start | %.2f CU/h | %.2f h elapsed' % (
            s['label'], s.get('gpu') or '-', s.get('start_time_iso'), s['start_balance'], s['rate_hourly'], elapsed))
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
    p.add_argument('--all', action='store_true', help='force-release every active assignment (needed when -s NAME has no recorded endpoints)')
    p.set_defaults(fn=cmd_stop)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == '__main__':
    sys.exit(main())
