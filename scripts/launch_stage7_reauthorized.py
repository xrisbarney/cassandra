"""One supervised fresh attempt per case, without overwriting earlier evidence."""

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from run_stage7_reauthorized import BASE, CASES, authority

from cyber_threat_modelling.data.store import write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--case', choices=CASES, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
record = authority(root)
base = root/BASE
base.mkdir(parents=True, exist_ok=True)
if (base/'fits'/args.case).exists():
    raise FileExistsError('Existing attempt is preserved; retries are prohibited.')
deadline = datetime.fromisoformat(record['global_deadline_utc'])
remaining = (deadline-datetime.now(UTC)).total_seconds()
if remaining <= 0:
    raise TimeoutError('Global sampling deadline reached.')
timeout = min(5400., remaining)
started = datetime.now(UTC).isoformat()
with (base/(args.case+'-fit.log')).open('x') as log:
    try:
        command = [sys.executable, '-u', str(root/'scripts/run_stage7_reauthorized.py'),
                   '--case', args.case]
        result = subprocess.run(command, cwd=root, stdout=log,
                                stderr=subprocess.STDOUT, timeout=timeout, check=False)
        execution = {'returncode': result.returncode, 'timed_out': False}
    except subprocess.TimeoutExpired:
        execution = {'returncode': None, 'timed_out': True}
write_json(base/(args.case+'-supervisor.json'), {
    **execution, 'case': args.case, 'timeout_seconds': timeout,
    'global_deadline_utc': deadline.isoformat(), 'attempt_count': 1,
    'started_utc': started, 'finished_utc': datetime.now(UTC).isoformat(),
    'partial_evidence_preserved': True, 'retry_permitted': False})
print(args.case, execution, flush=True)
if execution['timed_out'] or execution['returncode'] != 0:
    sys.exit(1)
