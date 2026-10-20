"""Build, freeze and verify revised-history panels from an immutable source snapshot."""

import importlib.metadata
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_config
from ..data.audit import verify_data_manifest
from ..data.store import sha256, utc_now, write_json
from ..provenance import git_revision, verify_freeze
from .incidents import incident_tables
from .panels import active_categories, cell_eligibility, select_cvss, topic_panel
from .topics import FrozenTopics


def coverage_plot(panel, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monthly = panel.groupby("month").agg(N=("N", "sum"), n_B=("n_B", "sum"), n_E=("n_E", "sum"))
    dates = pd.to_datetime(monthly.index)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, layout="constrained")
    axes[0].plot(dates, monthly.N.astype(float), color="#245c8b")
    axes[0].set_ylabel("Retained CVEs")
    axes[1].plot(dates, monthly.n_B / monthly.N, label="CVSS")
    e = monthly.n_E / monthly.N
    e.loc[monthly.index < "2021-04"] = np.nan
    axes[1].plot(dates, e, label="EPSS")
    axes[1].set_ylabel("Fraction with score")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend()
    unknown = panel[panel.category == panel.category.max()].set_index("month")
    axes[2].plot(dates, unknown.N.astype(float) / monthly.N, color="#905a33")
    axes[2].set_ylabel("Unknown fraction")
    for ax in axes:
        ax.axvline(pd.Timestamp("2019-01-01"), color="grey", linestyle="--", linewidth=1)
        ax.axvline(pd.Timestamp("2024-01-01"), color="grey", linestyle="--", linewidth=1)
        ax.grid(alpha=0.2)
    fig.suptitle("Revised-history panel coverage; no forecasts or evaluation scores")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def validate_panel(panel, records, settings):
    expected = len(pd.period_range(settings["panel_start"], settings["panel_end"], freq="M")) * (
        settings["learned_topics"] + 1
    )
    if len(panel) != expected or panel.duplicated(["month", "category"]).any():
        raise ValueError("Panel grid is incomplete or duplicated.")
    actual = panel.groupby("month").N.sum()
    counts = records.groupby("publication_month").size().reindex(actual.index, fill_value=0)
    complete = panel.groupby("month").m_N.all()
    if not np.array_equal(actual[complete].to_numpy(), counts[complete].to_numpy()):
        raise ValueError("Topic counts fail monthly conservation.")
    for channel, value in (("N", "N"), ("B", "B"), ("E", "E"), ("V", "V")):
        mask = panel["m_" + channel]
        if panel.loc[mask, value].isna().any() or panel.loc[~mask, value].notna().any():
            raise ValueError("Availability mask and target nulls disagree.")


def build_panels(root, source_manifest, panel_id, progress=print):
    root = Path(root).resolve()
    source_manifest = (root / source_manifest).resolve()
    if not source_manifest.is_relative_to(root):
        raise ValueError("Source manifest must be inside the project.")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", panel_id):
        raise ValueError("Invalid panel identifier.")
    output = root / "data/processed" / panel_id
    export = root / "docs/panels" / panel_id
    if output.exists() or export.exists():
        raise FileExistsError("Panel output already exists; use a new identifier.")
    config = load_config(root / "configs/pilot.toml")
    freeze = verify_freeze(root, config)
    source = verify_data_manifest(root, source_manifest)
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", source["snapshot"]):
        raise ValueError("Invalid upstream snapshot identifier.")
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    if not manifest["primary_source_ready_for_stage5"]:
        raise ValueError("Primary source audit is incomplete.")
    settings = json.loads((root / "configs/panel.json").read_text(encoding="utf-8"))
    if (
        settings["training_start"],
        settings["training_end"],
        settings["panel_start"],
        settings["panel_end"],
        settings["learned_topics"],
    ) != ("2010-01", "2018-12", "2010-01", "2025-12", 9):
        raise ValueError("Panel settings violate the frozen primary target definition.")
    policy_files = [
        "configs/panel.json",
        "configs/sec_review.csv",
        "configs/sec_sector_map.json",
        "docs/PANEL_DICTIONARY.md",
        "docs/STAGE2_FREEZE.json",
        "uv.lock",
    ]
    code_files = sorted((root / "src").rglob("*.py"))
    provenance = {
        "source_control": git_revision(root),
        "created_at": utc_now(),
        "policy_hashes": {p: sha256(root / p) for p in policy_files},
        "code_hashes": {p.relative_to(root).as_posix(): sha256(p) for p in code_files},
    }
    interim = root / "data/interim" / source["snapshot"]
    raw = pd.read_parquet(interim / "nvd_records.parquet")
    if raw.cve_id.duplicated().any():
        raise ValueError("Duplicate NVD identifier.")
    records = (
        raw[
            ~raw.rejected
            & raw.publication_month.between(settings["panel_start"], settings["panel_end"])
        ]
        .sort_values("cve_id")
        .copy()
    )
    progress("Fitting frozen TF-IDF/NMF on 2010-2018 descriptions only...", flush=True)
    started = time.perf_counter()
    model = FrozenTopics(settings).fit(records)
    fit_seconds = time.perf_counter() - started
    progress("Assigning CVEs with fixed vocabulary and components...", flush=True)
    started = time.perf_counter()
    labels, shares, reasons = model.transform(records.description_en)
    records["category"], records["topic_max_share"], records["assignment_reason"] = (
        labels,
        shares,
        reasons,
    )
    selected = pd.DataFrame(
        [select_cvss(p, settings["cvss_version_order"]) for p in records.cvss_json],
        index=records.index,
    )
    records = pd.concat([records, selected], axis=1)
    panel, kev = topic_panel(
        records,
        pd.read_parquet(interim / "epss_cohort_scores.parquet"),
        pd.read_parquet(interim / "kev_catalog.parquet"),
        pd.read_csv(interim / "nvd_monthly_coverage.csv"),
        pd.read_csv(interim / "epss_monthly_coverage.csv"),
        settings,
    )
    validate_panel(panel, records, settings)
    crosswalk = json.loads((root / "configs/sec_sector_map.json").read_text(encoding="utf-8"))
    links, first, sectors = incident_tables(
        pd.read_parquet(interim / "sec_candidates.parquet"),
        pd.read_csv(root / "configs/sec_review.csv", keep_default_na=False),
        crosswalk,
        pd.read_parquet(interim / "bea_61.parquet"),
        sorted(panel.month.unique()),
    )
    eligibility = pd.DataFrame(cell_eligibility(panel, sectors, sha256(source_manifest)))
    transform_seconds = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=False)
    export.mkdir(parents=True, exist_ok=False)
    model.save(output)
    # Verify portable reconstruction, including empty/OOV inputs, without refitting.
    reloaded = FrozenTopics.load(output, settings)
    probe = records.description_en.iloc[:100].tolist() + ["", "zzzxxyyqqq"]
    before, after = model.transform(probe), reloaded.transform(probe)
    np.testing.assert_array_equal(before[0], after[0])
    np.testing.assert_allclose(before[1], after[1], atol=1e-12)
    columns = [
        "cve_id",
        "publication_month",
        "category",
        "topic_max_share",
        "assignment_reason",
        "cvss",
        "cvss_version",
        "cvss_source",
        "cvss_vector",
        "source_hash",
        "metadata_available_at",
    ]
    records[columns].to_parquet(output / "cve_assignments.parquet", index=False)
    raw.loc[raw.rejected, ["cve_id", "publication_month", "status", "source_hash"]].to_parquet(
        output / "excluded_cves.parquet", index=False
    )
    for name, frame in (
        ("topic_month", panel),
        ("kev_assignments", kev),
        ("sec_filing_links", links),
        ("sec_incident_registrants", first),
        ("sector_month", sectors),
        ("cell_eligibility", eligibility),
    ):
        frame.to_parquet(output / f"{name}.parquet", index=False)
    strata = active_categories(panel, settings)
    topics = model.coherence()
    initial = records[records.publication_month <= settings["training_end"]]
    for item in topics:
        item["representative_initial_cves"] = (
            initial[initial.category == item["category"]]
            .sort_values(["topic_max_share", "cve_id"], ascending=[False, True])
            .head(3)
            .cve_id.tolist()
        )
    write_json(export / "topic_descriptions.json", topics)
    write_json(export / "category_strata.json", strata)
    panel.groupby("month").agg(
        N=("N", "sum"),
        n_B=("n_B", "sum"),
        n_E=("n_E", "sum"),
        observed_N_cells=("m_N", "sum"),
        observed_B_cells=("m_B", "sum"),
        observed_E_cells=("m_E", "sum"),
        V=("V", lambda x: x.sum(min_count=1)),
    ).to_csv(export / "monthly_coverage.csv")
    records.groupby(["publication_month", "cvss_version"]).size().rename("selected_scores").to_csv(
        export / "cvss_version_coverage.csv"
    )
    eligibility.groupby(["split", "horizon", "channel"]).agg(
        cells=("category", "size"),
        observed=("observed_flag", "sum"),
        target_eligible=("target_eligible", "sum"),
    ).to_csv(export / "eligibility_summary.csv")
    coverage_plot(panel, export / "coverage.png")
    summary = {
        "retained_cves": len(records),
        "excluded_rejected": int(raw.rejected.sum()),
        "training_cves": len(model.training_ids),
        "vocabulary_size": len(model.vectorizer.vocabulary_),
        "nmf_iterations": model.training_iterations,
        "topic_fit_seconds": fit_seconds,
        "transform_and_assembly_seconds": transform_seconds,
        "topic_month_cells": len(panel),
        "sector_month_cells": len(sectors),
        "observed_cells": {c: int(panel["m_" + c].sum()) for c in ("N", "B", "E", "V")},
        "unknown_cves": int((records.category == 10).sum()),
        "assignment_reasons": {
            str(k): int(v) for k, v in records.assignment_reason.value_counts().items()
        },
        "missing_selected_cvss": int(records.cvss.isna().sum()),
        "kev_additions_scored_calendar": int(panel.V.sum()),
        "kev_unknown_in_scored_calendar": int(panel.loc[panel.category == 10, "V"].sum()),
        "sec_reviewed_filing_registrant_links": len(links),
        "sec_first_in_sample_pairs": len(first),
        "sec_incident_groups": int(first.event_id.nunique()),
        "sec_observed_cells": 0,
        "active_categories": [r["category"] for r in strata if r["active"]],
    }
    write_json(export / "summary.json", summary)
    artifact_hashes = {
        p.relative_to(root).as_posix(): sha256(p)
        for directory in (output, export)
        for p in sorted(directory.iterdir())
        if p.is_file()
    }
    result = {
        "schema_version": 1,
        "panel_id": panel_id,
        "vintage_track": "revised",
        "source_data_manifest": source_manifest.relative_to(root).as_posix(),
        "source_data_manifest_sha256": sha256(source_manifest),
        "settings": settings,
        "protocol_freeze_sha256": freeze["sha256"],
        "artifacts": artifact_hashes,
        "software_versions": {
            n: importlib.metadata.version(n)
            for n in ("numpy", "scipy", "pandas", "pyarrow", "scikit-learn", "matplotlib")
        },
        "primary_count_panel_ready": True,
        "scoring_ready": False,
        "point_in_time_certified": False,
        "remaining_gates": [
            "Stage6_pre_scoring_execution_checks",
            "SEC_search_recall_population_and_first_disclosure_verification",
            "Stage11_economic_sector_crosswalk_and_target_compatibility",
        ],
        **provenance,
    }
    write_json(export / "panel_manifest.json", result)
    progress(json.dumps(summary, indent=2), flush=True)
    return export / "panel_manifest.json"


def verify_panels(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1 or manifest["vintage_track"] != "revised":
        raise ValueError("Unsupported panel schema/vintage.")

    def check(relative, expected):
        p = (root / relative).resolve()
        if not p.is_relative_to(root) or sha256(p) != expected:
            raise ValueError("Panel provenance path or artifact hash mismatch.")

    check(manifest["source_data_manifest"], manifest["source_data_manifest_sha256"])
    verify_data_manifest(root, root / manifest["source_data_manifest"])
    for field in ("artifacts", "policy_hashes"):
        for relative, digest in manifest[field].items():
            check(relative, digest)
    return {
        "status": "verified",
        "panel_id": manifest["panel_id"],
        "primary_count_panel_ready": manifest["primary_count_panel_ready"],
        "scoring_ready": manifest["scoring_ready"],
        "sha256": sha256(path),
    }
