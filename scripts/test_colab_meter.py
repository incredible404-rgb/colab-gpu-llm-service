"""colab_meter: the stop path must not pay for a pre-snapshot it cannot use. Every browser call is faked."""
import contextlib
import io
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import colab_meter as meter  # noqa: E402

CLEAN = {'assignmentsCount': 0, 'consumptionRateHourly': 0, 'currentBalance': 90.0, 'eligibleGpus': ['G4'], 'assignments': []}
LIVE = {'assignmentsCount': 1, 'consumptionRateHourly': 1.8, 'currentBalance': 100.0, 'eligibleGpus': ['G4'],
        'assignments': [{'endpoint': 'gpu-g4-s-1', 'accelerator': 'GPU', 'variant': 'G4', 'machineShape': 'standard'}]}


def session(label, endpoints, rate):
    return {'label': label, 'gpu': 'G4', 'start_time': time.time() - 3600, 'start_time_iso': 'x', 'start_balance': 100.0,
            'rate_hourly': rate, 'endpoints': endpoints}


class MeterStopTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patcher = mock.patch.object(meter, 'STATE_FILE', Path(tmp.name) / 'state.json')
        patcher.start()
        self.addCleanup(patcher.stop)

    def _stop(self, sessions, status, unassign, **kw):
        meter._save_state({'sessions': sessions})
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(meter.colab_sync, 'colab_status', return_value=status) as st, \
             mock.patch.object(meter.colab_sync, 'colab_unassign', return_value=unassign) as un, \
             contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = meter.stop_and_report(**kw)
        return rc, out.getvalue(), err.getvalue(), st, un

    def test_needs_backfill_truth_table(self):
        self.assertFalse(meter.needs_backfill({'sessions': []}))
        self.assertFalse(meter.needs_backfill({'sessions': [session('a', ['e'], 1.8), session('b', ['f'], 0.5)]}))
        self.assertTrue(meter.needs_backfill({'sessions': [session('a', [], 0)]}))
        self.assertTrue(meter.needs_backfill({'sessions': [session('a', ['e'], 0)]}))
        self.assertTrue(meter.needs_backfill({'sessions': [session('a', ['e'], 1.8), session('b', [], 1.0)]}))

    def test_complete_record_skips_the_presnapshot_round_trip(self):
        rc, out, _, status, unassign = self._stop(
            [session('g4', ['gpu-g4-s-1'], 1.8)], LIVE,
            {'before': LIVE, 'released': [{'endpoint': 'gpu-g4-s-1', 'ok': True, 'status': 200, 'step': 'POST'}], 'after': CLEAN},
            session='g4')
        self.assertEqual(rc, 0)
        status.assert_not_called()
        unassign.assert_called_once_with(['gpu-g4-s-1'])
        self.assertIn('unassign gpu-g4-s-1: OK', out)
        self.assertIn('CU bill for "g4"', out)
        self.assertIn('Verified: Active assignments: 0', out)
        self.assertEqual(meter._load_state()['sessions'], [])

    def test_session_recorded_during_allocation_is_still_backfilled_first(self):
        rc, out, _, status, unassign = self._stop(
            [session('g4', [], 0)], LIVE,
            {'before': LIVE, 'released': [{'endpoint': 'gpu-g4-s-1', 'ok': True, 'status': 200, 'step': 'POST'}], 'after': CLEAN},
            session='g4')
        self.assertEqual(rc, 0)
        status.assert_called_once()
        unassign.assert_called_once_with(['gpu-g4-s-1'])
        self.assertIn('track 2 (wall clock)', out)

    def test_waiting_record_without_a_live_assignment_still_refuses(self):
        rc, out, err, status, unassign = self._stop([session('g4', [], 0)], CLEAN, {}, session='g4')
        self.assertEqual(rc, 1)
        self.assertIn('no bound endpoints', err)
        status.assert_called_once()
        unassign.assert_not_called()

    def test_unknown_name_and_kill_all_unchanged(self):
        rc, _, err, status, unassign = self._stop([session('g4', ['gpu-g4-s-1'], 1.8)], LIVE, {}, session='nope')
        self.assertEqual(rc, 1)
        self.assertIn('No meter record named "nope"', err)
        status.assert_not_called()
        unassign.assert_not_called()
        rc, out, _, status, unassign = self._stop(
            [session('g4', ['gpu-g4-s-1'], 1.8)], LIVE, {'before': LIVE, 'released': [], 'after': CLEAN}, kill_all=True)
        self.assertEqual(rc, 0)
        status.assert_not_called()
        unassign.assert_called_once_with(None)


class MeterStateTests(unittest.TestCase):
    def test_state_roundtrip_and_cleaning(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 's.json'
            with mock.patch.object(meter, 'STATE_FILE', path):
                self.assertEqual(meter._load_state(), {'sessions': []})
                meter._save_state({'sessions': [session('a', ['e'], 1.0), {'label': 5}]})
                loaded = meter._load_state()
                self.assertEqual([s['label'] for s in loaded['sessions']], ['a'])
                path.write_text('{broken', encoding='utf-8')
                self.assertEqual(meter._load_state(), {'sessions': []})
                self.assertFalse(path.with_name(path.name + '.tmp').exists())
                json.loads(json.dumps(loaded))


if __name__ == '__main__':
    unittest.main()
