"""Validate experiments, prepare run records, and audit source data."""

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
    prepare.add_argument("--data-manifest", type=Path)
    audit = commands.add_parser("audit-data", help="Download raw sources and audit coverage")
    audit.add_argument("--snapshot", required=True)
    audit.add_argument("--credentials-file", type=Path)
    audit.add_argument(
        "--sources", nargs="+", choices=["nvd", "history", "epss", "kev", "sec", "bea", "events"]
    )
    audit.add_argument("--finalize-only", action="store_true")
    audit.add_argument("--finalize", action="store_true")
    check_data = commands.add_parser(
        "verify-data", help="Verify source and derived artifact hashes"
    )
    check_data.add_argument("manifest", type=Path)
    panel = commands.add_parser("build-panels", help="Freeze initial topics and monthly panels")
    panel.add_argument("--data-manifest", required=True, type=Path)
    panel.add_argument("--panel-id", required=True)
    check_panel = commands.add_parser("verify-panels", help="Verify panel and upstream hashes")
    check_panel.add_argument("manifest", type=Path)
    forecasts = commands.add_parser(
        "forecast-development", help="Save development-only control forecasts"
    )
    forecasts.add_argument("settings", type=Path)
    forecasts.add_argument("--panel-manifest", type=Path, required=True)
    forecasts.add_argument("--output", type=Path, required=True)
    scores = commands.add_parser("score-development", help="Score saved development forecasts")
    scores.add_argument("run", type=Path)
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    try:
        if args.command == "forecast-development":
            from .evaluation.backtest import forecast_development

            settings = json.loads((root / args.settings).read_text(encoding="utf-8"))
            print(
                forecast_development(root, settings, root / args.panel_manifest, root / args.output)
            )
            return 0
        if args.command == "score-development":
            from .evaluation.backtest import score_development

            print(score_development(root, root / args.run))
            return 0
        if args.command == "build-panels":
            from .features.build import build_panels

            print(build_panels(root, args.data_manifest, args.panel_id))
            return 0
        if args.command == "verify-panels":
            from .features.build import verify_panels

            print(json.dumps(verify_panels(root, root / args.manifest), indent=2))
            return 0
        if args.command == "audit-data":
            from .data.audit import COMPONENTS, finalize_audit, run_audit

            verify_freeze(root, load_config(root / "configs" / "pilot.toml"))
            if not args.finalize_only:
                run_audit(root, args.snapshot, args.credentials_file, args.sources or COMPONENTS)
            if args.finalize or args.finalize_only:
                print(finalize_audit(root, args.snapshot))
            return 0
        if args.command == "verify-data":
            from .data.audit import verify_data_manifest

            print(json.dumps(verify_data_manifest(root, root / args.manifest), indent=2))
            return 0
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
            destination = prepare_run(
                root,
                config,
                root / args.output,
                root / args.data_manifest if args.data_manifest else None,
            )
            print(f"Prepared {destination}; no data ingested and no model fitted.")
    except (ConfigurationError, OSError, ValueError, KeyError) as exc:
        # Configuration validation messages never include unknown values or environment data.
        parser.exit(2, f"Error: {exc}\n")
    return 0
