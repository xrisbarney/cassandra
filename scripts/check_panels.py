"""Independently reconcile saved monthly panels to the Stage 4 normalized records."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cyber_threat_modelling.features.build import verify_panels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / args.manifest).read_text(encoding="utf-8"))
    verify_panels(root, root / args.manifest)
    source = json.loads((root / manifest["source_data_manifest"]).read_text(encoding="utf-8"))
    interim = root / "data/interim" / source["snapshot"]
    processed = root / "data/processed" / manifest["panel_id"]
    raw = pd.read_parquet(interim / "nvd_records.parquet")
    raw = raw[~raw.rejected]
    assignments = pd.read_parquet(processed / "cve_assignments.parquet")
    panel = pd.read_parquet(processed / "topic_month.parquet").set_index(["month", "category"])
    assert not assignments.cve_id.duplicated().any()
    assert set(assignments.cve_id) == set(raw.cve_id)
    grouped = assignments.groupby(["publication_month", "category"])
    np.testing.assert_array_equal(panel.N, grouped.size().reindex(panel.index, fill_value=0))
    selected = []
    for row in raw.itertuples():
        scores = json.loads(row.cvss_json)
        if not scores:
            selected.append((row.cve_id, np.nan))
            continue
        scores.sort(
            key=lambda e: (
                {"4.0": 0, "3.1": 1, "3.0": 2, "2.0": 3}[e["version"]],
                (0 if e["source"] == "nvd@nist.gov" else 1)
                if e["type"] == "Primary"
                else (2 if e["source"] == "nvd@nist.gov" else 3),
                e["source"],
                e["vector"],
            )
        )
        selected.append((row.cve_id, scores[0]["base_score"]))
    raw_scores = pd.DataFrame(selected, columns=["cve_id", "expected"])
    joined = assignments.merge(raw_scores, on="cve_id", validate="one_to_one")
    np.testing.assert_allclose(joined.cvss, joined.expected, equal_nan=True)
    severity = joined.groupby(["publication_month", "category"]).expected
    np.testing.assert_array_equal(panel.n_B, severity.count().reindex(panel.index, fill_value=0))
    np.testing.assert_allclose(panel.B, severity.mean().reindex(panel.index), equal_nan=True)
    epss = pd.read_parquet(interim / "epss_cohort_scores.parquet").merge(
        assignments[["cve_id", "publication_month", "category"]], on="cve_id", validate="one_to_one"
    )
    assert epss.month.eq(epss.publication_month).all()
    egroup = epss.groupby(["month", "category"]).epss
    np.testing.assert_array_equal(panel.n_E, egroup.count().reindex(panel.index, fill_value=0))
    np.testing.assert_allclose(panel.E, egroup.mean().reindex(panel.index), equal_nan=True)
    kev = pd.read_parquet(interim / "kev_catalog.parquet")
    additions = kev[kev.addition_month.between("2021-12", "2025-12")].copy()
    additions["category"] = (
        additions.cveID.map(assignments.set_index("cve_id").category).fillna(10).astype(int)
    )
    expected_v = additions.groupby(["addition_month", "category"]).size()
    observed = panel[panel.m_V]
    np.testing.assert_array_equal(observed.V, expected_v.reindex(observed.index, fill_value=0))
    training_ids = json.loads((processed / "topic_training_ids.json").read_text(encoding="utf-8"))
    assert training_ids == sorted(
        raw.loc[raw.publication_month.between("2010-01", "2018-12"), "cve_id"]
    )
    strata = json.loads(
        (root / "docs/panels" / manifest["panel_id"] / "category_strata.json").read_text(
            encoding="utf-8"
        )
    )
    for row in strata:
        n = panel.loc[(slice("2010-01", "2018-12"), row["category"]), "N"]
        assert row["initial_assignments"] == n.sum()
        assert row["active"] == bool(n.sum() >= 50 and (n > 0).sum() >= 24)
    sectors = pd.read_parquet(processed / "sector_month.parquet")
    assert not sectors.m_D.any() and sectors.D.isna().all()
    assert sectors.eligible_registrants.isna().all()
    first = pd.read_parquet(processed / "sec_incident_registrants.parquet")
    assert not first.duplicated(["event_id", "cik"]).any()
    assert sectors.reviewed_pairs_first_in_sample.sum() == len(first)
    eligibility = pd.read_parquet(processed / "cell_eligibility.parquet")
    assert not eligibility.eligible_for_scoring.any()
    assert eligibility["available_at"].eq("unknown").all()
    assert not eligibility.duplicated(
        ["split", "origin_month", "horizon", "channel", "category"]
    ).any()
    for split, expected in (("development", [60, 58, 55, 49]), ("evaluation", [24, 22, 19, 13])):
        for h, count in zip([1, 3, 6, 12], expected, strict=True):
            n = eligibility[
                (eligibility.split == split)
                & (eligibility.horizon == h)
                & (eligibility.channel == "N")
            ]
            assert len(n) == count * 10 and n.target_eligible.all()
    report = {
        "status": "passed",
        "retained_cves": len(assignments),
        "training_ids": len(training_ids),
        "topic_month_cells": len(panel),
        "epss_cohort_scores": len(epss),
        "kev_additions": len(additions),
        "incident_registrant_pairs_in_sample": len(first),
        "cell_eligibility_rows": len(eligibility),
        "source_control_at_build": manifest["source_control"],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
