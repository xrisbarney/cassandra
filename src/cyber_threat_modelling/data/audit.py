"""Stage 4 orchestration, source coverage, and hash-bound audit manifests."""

import importlib.metadata
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from .economics import ingest_bea, ingest_events
from .history import audit_history
from .nvd import ingest_nvd
from .sec import ingest_sec
from .signals import ingest_epss, ingest_kev
from .store import RawStore, credentials, sha256, utc_now, write_json

COMPONENTS = ("nvd", "history", "epss", "kev", "sec", "bea", "events")


def run_audit(root, snapshot, credential_path=None, components=COMPONENTS):
    import pandas as pd

    root = Path(root).resolve()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", snapshot):
        raise ValueError("Snapshot identifier must be a simple directory name.")
    if (root / "docs" / "audits" / snapshot / "data_manifest.json").exists():
        raise FileExistsError("Audit is frozen; use verify-data or a new snapshot identifier.")
    output = root / "data" / "interim" / snapshot
    output.mkdir(parents=True, exist_ok=True)
    secrets = credentials(credential_path)
    nvd = None
    for component in components:
        store = RawStore(root / "data" / "raw" / snapshot)
        if component == "nvd":
            result, nvd = ingest_nvd(store, output, last_year=datetime.now(timezone.utc).year)
        elif component == "history":
            result = audit_history(store, secrets.get("NVD_API_KEY"))
        elif component in ("epss", "kev"):
            if nvd is None:
                nvd = pd.read_parquet(output / "nvd_records.parquet")
            result = {"epss": ingest_epss, "kev": ingest_kev}[component](store, output, nvd)
        elif component == "sec":
            result = ingest_sec(store, output, secrets.get("SEC_USER_AGENT"))
        elif component == "bea":
            result = ingest_bea(store, output, secrets.get("BEA_API_KEY"))
        elif component == "events":
            result = ingest_events(store, output, root / "configs" / "event_evidence.json")
        else:
            raise ValueError("Unsupported source component.")
        write_json(output / f"{component}_audit.json", result)
        write_json(output / f"{component}_sources.json", list(store.records.values()))
        print(f"{component}: audit saved", flush=True)
    return output


def eligibility_rows(reports):
    """Calendar/source gates; topic-level observed cells cannot be certified before Stage 5."""
    import pandas as pd

    nvd = {r["month"]: r for r in reports["nvd"]["monthly"]}
    epss = {r["month"]: r for r in reports["epss"]["months"]}
    for split, start, end in (
        ("development", "2019-01", "2023-12"),
        ("evaluation", "2024-01", "2025-12"),
    ):
        first, last = pd.Period(start, freq="M"), pd.Period(end, freq="M")
        for horizon in (1, 3, 6, 12):
            for origin in pd.period_range(first - 1, last - horizon, freq="M"):
                target = str(origin + horizon)
                for channel in ("N", "B", "E", "V", "D"):
                    available = nvd[target]["feed_coverage_verified"]
                    if channel == "B":
                        available &= (
                            nvd[target]["nonrejected_records"] > nvd[target]["missing_cvss"]
                        )
                    if channel == "E":
                        available &= epss.get(target, {}).get("snapshot_available", False)
                    if channel == "V":
                        available &= target >= "2021-12"
                    if channel == "D":
                        available = False
                    yield {
                        "split": split,
                        "origin_month": str(origin),
                        "target_month": target,
                        "horizon": horizon,
                        "channel": channel,
                        "category": "unassigned_stage5",
                        "vintage_track": "revised",
                        "source_available": bool(available),
                        "available_at": "unknown",
                        "observed_flag": False,
                        "eligible_for_scoring": False,
                        "eligibility_reason": "stage5_topic_or_sector_panel_pending"
                        if available
                        else "source_or_channel_audit_gate_not_met",
                    }


def finalize_audit(root, snapshot):
    import pandas as pd

    root = Path(root).resolve()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", snapshot):
        raise ValueError("Invalid snapshot identifier.")
    output = root / "data" / "interim" / snapshot
    export = root / "docs" / "audits" / snapshot
    if (export / "data_manifest.json").exists():
        raise FileExistsError("Audit manifest is already frozen; use a new snapshot identifier.")
    export.mkdir(parents=True, exist_ok=True)
    reports, sources = {}, []
    for component in COMPONENTS:
        reports[component] = json.loads(
            (output / f"{component}_audit.json").read_text(encoding="utf-8")
        )
        records = json.loads((output / f"{component}_sources.json").read_text(encoding="utf-8"))
        for record in records:
            record["publication_date"] = record.get("publication_date", "unknown")
            path = root / "data" / "raw" / snapshot / record["source_id"]
            if record["http_status"] != 200:
                path = path.with_name(path.name + ".http-error")
            record["path"] = path.relative_to(root).as_posix()
            if sha256(path) != record["sha256"]:
                raise ValueError("Raw source hash changed during audit.")
            sources.append(record)
    # Export compact aggregate coverage and source hashes, never raw responses/credentials.
    source_manifest = {"schema_version": 1, "snapshot": snapshot, "records": sources}
    write_json(export / "source_manifest.json", source_manifest)
    write_json(export / "coverage_summary.json", reports)
    pd.DataFrame(eligibility_rows(reports)).to_csv(export / "source_eligibility.csv", index=False)
    for name in (
        "nvd_monthly_coverage.csv",
        "epss_monthly_coverage.csv",
        "sec_search_coverage.csv",
    ):
        (export / name).write_bytes((output / name).read_bytes())
    artifacts = {
        p.relative_to(root).as_posix(): sha256(p) for p in sorted(output.iterdir()) if p.is_file()
    }
    for p in sorted(export.iterdir()):
        if p.is_file() and p.name != "data_manifest.json":
            artifacts[p.relative_to(root).as_posix()] = sha256(p)
    from ..provenance import git_revision

    manifest = {
        "schema_version": 1,
        "snapshot": snapshot,
        "created_at": utc_now(),
        "vintage_track": "revised",
        "publication_window": ["2010-01", "2025-12"],
        "source_manifest": (export / "source_manifest.json").relative_to(root).as_posix(),
        "source_manifest_sha256": sha256(export / "source_manifest.json"),
        "artifacts": artifacts,
        "source_control": git_revision(root),
        "python": sys.version,
        "software_versions": {
            name: importlib.metadata.version(name) for name in ("pandas", "numpy", "pyarrow")
        },
        "dependency_lock_sha256": sha256(root / "uv.lock"),
        "ingestion_source_sha256": {
            p.relative_to(root).as_posix(): sha256(p)
            for p in sorted((root / "src" / "cyber_threat_modelling" / "data").glob("*.py"))
        },
        "policy_sha256": sha256(root / "docs" / "DATA_DICTIONARY.md"),
        "event_register_sha256": sha256(root / "configs" / "event_evidence.json"),
        "protocol_freeze_sha256": sha256(root / "docs" / "STAGE2_FREEZE.json"),
        "primary_source_ready_for_stage5": reports["nvd"]["all_feeds_verified"],
        "scoring_ready": False,
        "point_in_time_certified": False,
        "remaining_gates": [
            "Stage5_topic_panel_and_cell_masks",
            "SEC_incident_adjudication",
            "economic_target_compatibility",
            "pre_scoring_execution_checks",
        ],
    }
    write_json(export / "data_manifest.json", manifest)
    return export / "data_manifest.json"


def verify_data_manifest(root, path, vintage="revised"):
    root, path = Path(root).resolve(), Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1 or manifest["vintage_track"] != vintage:
        raise ValueError("Data manifest schema/vintage does not match this run.")

    def check(relative, expected):
        candidate = (root / relative).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError("Data manifest path escapes the project.")
        if sha256(candidate) != expected:
            raise ValueError("Data manifest artifact hash mismatch.")

    check(manifest["source_manifest"], manifest["source_manifest_sha256"])
    sources = json.loads((root / manifest["source_manifest"]).read_text(encoding="utf-8"))
    for record in sources["records"]:
        check(record["path"], record["sha256"])
    for relative, digest in manifest["artifacts"].items():
        check(relative, digest)
    check("docs/DATA_DICTIONARY.md", manifest["policy_sha256"])
    check("configs/event_evidence.json", manifest["event_register_sha256"])
    check("docs/STAGE2_FREEZE.json", manifest["protocol_freeze_sha256"])
    return {
        "path": str(path),
        "sha256": sha256(path),
        "snapshot": manifest["snapshot"],
        "vintage_track": vintage,
        "scoring_ready": manifest["scoring_ready"],
    }
