"""Run all predeclared v2 posterior sensitivity cases, preserving every result."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from cyber_threat_modelling.models.stage7_priors import CANDIDATES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    for candidate in CANDIDATES:
        if candidate == 'primary':
            continue
        result = subprocess.run([
            sys.executable, '-u', 'scripts/stage7_batch.py', '--root', str(args.root/candidate),
            '--workers', '3', '--anchor-factors', '--dense-mass', '--prior-candidate', candidate,
            '--include', 'factors_sparse', 'representative', 'fallback_zero', '--seeds', '7101',
        ], check=False)
        print(json.dumps({'candidate': candidate, 'coordinator_exit': result.returncode}),
              flush=True)


if __name__ == '__main__':
    main()
