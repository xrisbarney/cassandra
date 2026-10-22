"""Enforce per-attempt/global wall-clock limits outside the sampler process."""

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from run_stage7_terminal import DEADLINE, authority

from cyber_threat_modelling.data.store import write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--case', choices=['joint_free', 'joint_fixed'], required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
authority(root)
base = root/'artifacts/stage7/terminal-v1'
folder = base/'fits'/args.case
if folder.exists():
    raise FileExistsError('Existing attempt is preserved; retries are prohibited.')
remaining = (DEADLINE-datetime.now(UTC)).total_seconds()
if remaining <= 0:
    raise TimeoutError('Global convergence deadline reached.')
timeout = min(1200., remaining)
with (base/(args.case+'-fit.log')).open('x') as log:
    try:
        result = subprocess.run([sys.executable, '-u', str(root/'scripts/run_stage7_terminal.py'),
                                 '--case', args.case], cwd=root, stdout=log,
                                stderr=subprocess.STDOUT, timeout=timeout, check=False)
        execution = {'returncode': result.returncode, 'timed_out': False}
    except subprocess.TimeoutExpired:
        execution = {'returncode': None, 'timed_out': True}
write_json(base/(args.case+'-supervisor.json'), {
    **execution, 'case': args.case, 'timeout_seconds': timeout,
    'global_deadline_utc': DEADLINE.isoformat(), 'attempt_count': 1,
    'finished_utc': datetime.now(UTC).isoformat(),
    'partial_evidence_preserved': True, 'retry_permitted': False})
print(args.case, execution, flush=True)
if execution['timed_out'] or execution['returncode'] != 0:
    sys.exit(1)
