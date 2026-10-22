"""Run every test file in a fresh process and bind the result to unchanged source."""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.provenance import file_hash, git_revision


def source_snapshot():
    paths = [Path('uv.lock'), *Path('src').rglob('*.py'),
             *Path('tests').rglob('*.py'), *Path('scripts').glob('*.py')]
    return {p.as_posix(): file_hash(p) for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = sorted(Path('tests').glob('test_*.py'))
    source_hashes = source_snapshot()
    start_revision = git_revision(Path.cwd())
    collect_log = args.output/'collection.log'
    with collect_log.open('xb') as stream:
        code = subprocess.run([sys.executable, '-u', '-m', 'pytest', '--collect-only', '-q'],
                              stdout=stream, stderr=subprocess.STDOUT).returncode
    collected = re.findall(r'(\d+) tests? collected', collect_log.read_text(encoding='utf-8'))
    if code or not collected:
        raise RuntimeError('Test collection failed; inspect collection.log.')
    expected = int(collected[-1])
    write_json(args.output/'run.json', {'source_control': start_revision,
                                       'source_sha256': source_hashes,
                                       'expected_tests': expected,
                                       'collection_sha256': file_hash(collect_log)})
    records = []
    for path in paths:
        log = args.output/f'{path.stem}.log'
        started = time.perf_counter()
        with log.open('xb') as stream:
            code = subprocess.run([sys.executable, '-u', '-m', 'pytest', '-q', str(path)],
                                  stdout=stream, stderr=subprocess.STDOUT).returncode
        passed = re.findall(r'(\d+) passed', log.read_text(encoding='utf-8'))
        row = {'file': path.as_posix(), 'sha256': file_hash(path), 'exit_code': code,
               'passed': int(passed[-1]) if passed else 0,
               'seconds': time.perf_counter()-started, 'log': log.as_posix(),
               'log_sha256': file_hash(log)}
        records.append(row)
        write_json(args.output/'progress.json', records)
        print(json.dumps(row), flush=True)
    end_hashes = source_snapshot()
    unchanged = source_hashes == end_hashes
    total = sum(r['passed'] for r in records)
    passed = all(r['exit_code'] == 0 for r in records) and total == expected
    result = {'source_control_start': start_revision,
              'source_control_end': git_revision(Path.cwd()),
              'source_sha256': source_hashes, 'source_unchanged': unchanged,
              'source_sha256_end': end_hashes,
              'expected_tests': expected, 'passed': total, 'files': len(records),
              'all_passed': passed and unchanged, 'records': records,
              'mode': 'Every collected test file in a fresh process; no deselection.',
              'collection_sha256': file_hash(collect_log)}
    write_json(args.output/'result.json', result)
    if not result['all_passed']:
        raise RuntimeError('Tests failed, collection differs, or source changed during checks.')


if __name__ == '__main__':
    main()
