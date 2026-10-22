"""Separate numerical gates from documented unresolved posterior exploration."""

from pathlib import Path

from ..provenance import file_hash


def recovery_eligibility(folder, status, routing, root):
    root, folder = Path(root).resolve(), Path(folder).resolve()
    matches = []
    for item in routing.get('recovery_exclusions', []):
        target = Path(item['directory']).resolve()
        evidence = Path(item['evidence']).resolve()
        if not target.is_relative_to(root) or not evidence.is_relative_to(root):
            raise ValueError('Recovery exclusion escapes the project.')
        if not item['reason'].strip() or file_hash(evidence) != item['evidence_sha256']:
            raise ValueError('Recovery exclusion requires unchanged evidence and a reason.')
        if file_hash(target/'result.json') != item['result_sha256']:
            raise ValueError('Recovery exclusion refers to a changed result.')
        if target == folder:
            matches.append(item)
    if len(matches) > 1:
        raise ValueError('Duplicate recovery exclusions.')
    return {'recovery_eligible': status == 'accepted' and not matches,
            'recovery_exclusion': matches[0] if matches else None}
