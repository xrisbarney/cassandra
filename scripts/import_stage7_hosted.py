"""Verify a terminal hosted case before publishing its local result marker."""

import argparse
import json
import os
import shutil
from pathlib import Path

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.models.stage7_priors import CURRENT_PRIOR_ID, verify_activation
from cyber_threat_modelling.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--head', required=True)
    parser.add_argument('--artifact-id', required=True)
    parser.add_argument('--artifact-digest', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    if not args.download.resolve().is_relative_to(root):
        raise ValueError('Downloaded case must be inside the project.')
    if args.output.exists() or not args.output.resolve().is_relative_to(root):
        raise ValueError('Import audit must be a fresh project directory.')
    from audit_stage7_campaign import checked_run
    from ci_stage7 import PLAN, checked_plan

    plan = checked_plan(root, fitting=True)
    matches = [c for c in plan['cases'] if c['key'] == args.case]
    if len(matches) != 1:
        raise ValueError('Case is outside the declared hosted grid.')
    case = matches[0]
    destination = root / case['directory']
    relative = Path(case['directory']).relative_to('artifacts/stage7')
    folder = args.download / relative
    if not (folder / 'result.json').is_file():
        raise ValueError('Downloaded case is not terminal; no local result may be published.')
    completion_paths = sorted((args.download / 'hosted-audits' / args.case).glob(
        'attempt*/completion.json'))
    if not completion_paths:
        raise ValueError('Missing hosted completion audit.')
    completion = json.loads(completion_paths[-1].read_text(encoding='utf-8'))
    host = completion['environment']
    if (not completion['case_is_terminal'] or host['github_run_id'] != args.run_id
            or host['github_sha'] != args.head or host['numba_opt'] != plan['numba_opt']
            or completion['plan_sha256'] != file_hash(root / PLAN)):
        raise ValueError('Hosted completion identity does not match the declared execution.')
    if completion['case']['result_sha256'] != file_hash(folder / 'result.json'):
        raise ValueError('Result differs from the hosted diagnostic review.')
    run = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
    if run['source_control']['revision'] != args.head or run['source_control']['dirty']:
        raise ValueError('Hosted source revision is different or dirty.')
    for name, digest in run['source_sha256'].items():
        source = (root / name).resolve()
        if not source.is_relative_to(root) or file_hash(source) != digest:
            raise ValueError(f'Hosted source differs from declared local source: {name}')
    for name in ('data', 'truth', 'truth_original_coordinates'):
        if file_hash(folder / f'{name}.npz') != case[name + '_sha256']:
            raise ValueError('Hosted fit used a different canonical synthetic fixture.')
    result = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
    expected = {**result['artifacts'], 'result.json': file_hash(folder / 'result.json')}
    if {p.name for p in folder.iterdir()} != set(expected):
        raise ValueError('Unexpected or missing file in downloaded terminal case.')
    for name, digest in expected.items():
        path = (folder / name).resolve()
        if (not path.is_relative_to(folder.resolve()) or path.is_symlink()
                or not path.is_file() or file_hash(path) != digest):
            raise ValueError('Invalid downloaded case artifact.')
    reservation = destination / 'remote_reservation.json'
    reserved = json.loads(reservation.read_text(encoding='utf-8'))
    if (reserved['key'] != args.case or reserved['expected_head'] != args.head
            or reserved['plan_sha256'] != file_hash(root / PLAN)):
        raise ValueError('Local reservation belongs to another execution.')
    for path in destination.iterdir():
        if path == reservation:
            continue
        if (path.name not in expected or not path.is_file() or path.is_symlink()
                or file_hash(path) != expected[path.name]):
            raise ValueError('Local destination contains conflicting evidence; preserve it.')
    activation = verify_activation(root, CURRENT_PRIOR_ID)
    reviewed = checked_run(folder, case['candidate'], case['scenario'], case['seed'],
                           activation, recompute=True)
    args.output.mkdir(parents=True, exist_ok=False)
    # Existing matching files support an interrupted copy; conflicting files are refused.
    # Publish result.json last, only after every referenced artifact verifies locally.
    for name in sorted(set(expected) - {'result.json'}):
        path = destination / name
        if not path.exists():
            shutil.copy2(folder / name, path)
        if file_hash(path) != expected[name]:
            raise ValueError('Transferred artifact differs from the verified download.')
    marker = destination / 'result.json'
    if not marker.exists():
        temporary = args.output / 'result.json.transfer'
        shutil.copy2(folder / 'result.json', temporary)
        if file_hash(temporary) != expected['result.json']:
            raise ValueError('Transferred result marker differs from the verified download.')
        os.replace(temporary, marker)
    write_json(args.output / 'audit.json', {
        'case': args.case, 'status': reviewed['status'], 'directory': destination.as_posix(),
        'result_sha256': expected['result.json'], 'artifact_id': args.artifact_id,
        'provider_artifact_archive_digest': args.artifact_digest,
        'archive_digest_note': 'Provider metadata; extracted file hashes verified locally.',
        'github_run_id': args.run_id, 'github_sha': args.head,
        'reservation_sha256': file_hash(reservation), 'plan_sha256': file_hash(root / PLAN),
        'download': args.download.as_posix(), 'script_sha256': file_hash(Path(__file__)),
        'completion_audits': {p.as_posix(): file_hash(p) for p in completion_paths},
        'review': reviewed, 'real_data_read': False, 'held_out_scoring_enabled': False,
    })
    print(args.case, reviewed['status'], 'verified and imported', flush=True)


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        main()
