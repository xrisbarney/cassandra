"""Stage 3 commands: validate configuration and prepare an unexecuted run."""

import argparse
import json
from pathlib import Path

from .config import ConfigurationError, load_config
from .provenance import prepare_run, verify_freeze


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Check settings and frozen protocol hashes")
    validate.add_argument("config", type=Path)
    prepare = commands.add_parser("prepare-run", help="Record provenance; does not fit or ingest")
    prepare.add_argument("config", type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    try:
        config = load_config(root / args.config)
        verify_freeze(root, config)
        if args.command == "validate":
            print(
                json.dumps(
                    {
                        "status": "valid",
                        "configuration_sha256": config.sha256,
                        "evaluation_origin_counts": config.origin_counts(),
                        "development_origin_counts": config.origin_counts("development"),
                    },
                    indent=2,
                )
            )
        else:
            destination = prepare_run(root, config, root / args.output)
            print(f"Prepared {destination}; no data ingested and no model fitted.")
    except (ConfigurationError, OSError, ValueError, KeyError) as exc:
        # Configuration validation messages never include unknown values or environment data.
        parser.exit(2, f"Error: {exc}\n")
    return 0
