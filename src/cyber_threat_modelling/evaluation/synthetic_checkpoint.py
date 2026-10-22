"""Seal only completed failed attempts; never resume a partially sampled chain."""

import json

from cyber_threat_modelling.data.store import write_json
from cyber_threat_modelling.provenance import file_hash


def inventory(folder, exclude=()):
    files = {}
    for path in folder.iterdir():
        if path.name in exclude:
            continue
        if not path.is_file() or path.is_symlink():
            raise ValueError('Checkpoint contains an unexpected directory or link.')
        files[path.name] = file_hash(path)
    return files


def seal(folder, attempts):
    retry = len(attempts) - 1
    if retry not in (0, 1) or any(a['accepted'] for a in attempts):
        raise ValueError('Only a completed unsuccessful nonfinal attempt can be paused.')
    path = folder / f'checkpoint_{retry}.json'
    if path.exists() or (folder / 'result.json').exists():
        raise ValueError('Checkpoint or terminal result already exists.')
    write_json(path, {'next_retry': retry + 1, 'artifacts': inventory(folder),
                      'interpretation': 'Paused between independent prescribed attempts; '
                      'not a terminal result and not a chain continuation.'})


def restore(folder, root):
    if (folder / 'result.json').exists():
        raise ValueError('A terminal case cannot be resumed.')
    paths = sorted(folder.glob('checkpoint_*.json'))
    if not paths:
        raise ValueError('No sealed attempt checkpoint.')
    path = paths[-1]
    record = json.loads(path.read_text(encoding='utf-8'))
    next_retry = record['next_retry']
    if next_retry not in (1, 2) or path.name != f'checkpoint_{next_retry - 1}.json':
        raise ValueError('Unexpected checkpoint retry.')
    if inventory(folder, exclude=(path.name,)) != record['artifacts']:
        raise ValueError('Changed, extra or missing checkpoint artifact.')
    run = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
    for name, digest in run['source_sha256'].items():
        source = (root / name).resolve()
        if not source.is_relative_to(root.resolve()) or file_hash(source) != digest:
            raise ValueError(f'Changed checkpoint source: {name}')
    attempts = []
    budgets = [(1000, 1000, .95, 12), (2000, 1000, .99, 15)]
    for retry in range(next_retry):
        attempt = json.loads((folder / f'attempt_{retry}.json').read_text(encoding='utf-8'))
        actual = tuple(attempt[k] for k in ('warmup', 'draws', 'target_accept', 'max_treedepth'))
        if attempt['retry'] != retry or attempt['accepted'] or attempt['chains'] != 4:
            raise ValueError('Checkpoint attempt is not a completed unsuccessful attempt.')
        if actual != budgets[retry]:
            raise ValueError('Checkpoint sampling budget changed.')
        if not attempt.get('early_failure') and not (folder / f'posterior_{retry}.nc').exists():
            raise ValueError('Completed attempt is missing its posterior.')
        attempts.append(attempt)
    return run, attempts, file_hash(path)
