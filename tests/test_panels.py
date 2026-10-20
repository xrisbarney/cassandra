import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cyber_threat_modelling.features.build import build_panels, validate_panel, verify_panels
from cyber_threat_modelling.features.incidents import incident_tables
from cyber_threat_modelling.features.panels import (
    active_categories,
    cell_eligibility,
    select_cvss,
    topic_panel,
)
from cyber_threat_modelling.features.topics import FrozenTopics, hard_assign


@pytest.fixture
def settings():
    return json.loads((Path(__file__).parents[1] / "configs/panel.json").read_text())


def test_topics_never_fit_future_or_rejected_and_reload(tmp_path, settings):
    settings.update(learned_topics=2, transform_batch_size=2)
    settings["vectorizer"].update(min_df=1, max_df=1.0, stop_words=None)
    records = pd.DataFrame(
        {
            "cve_id": list("abcdef"),
            "publication_month": ["2010-01"] * 4 + ["2024-01", "2010-01"],
            "rejected": [False] * 5 + [True],
            "description_en": [
                "alpha memory buffer",
                "alpha memory overflow",
                "beta web script",
                "beta web injection",
                "futuretoken",
                "rejectedtoken",
            ],
        }
    )
    model = FrozenTopics(settings).fit(records)
    assert model.training_ids == list("abcd")
    assert "futuretoken" not in model.vectorizer.vocabulary_
    assert "rejectedtoken" not in model.vectorizer.vocabulary_
    before = model.nmf.components_.copy()
    labels, _, reasons = model.transform(["", "futuretoken", "alpha memory buffer"])
    assert list(labels)[:2] == [3, 3]
    assert reasons[:2] == ["empty_text", "no_vocabulary_terms"]
    np.testing.assert_array_equal(before, model.nmf.components_)
    model.save(tmp_path)
    restored = FrozenTopics.load(tmp_path, settings)
    np.testing.assert_array_equal(
        restored.transform(records.description_en)[0], model.transform(records.description_en)[0]
    )
    changed = records.copy()
    changed.loc[4, "description_en"] = "different future text"
    refitted = FrozenTopics(settings).fit(changed)
    np.testing.assert_array_equal(before, refitted.nmf.components_)


def test_topic_ties_and_unknown():
    np.testing.assert_array_equal(
        hard_assign(np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 1.0]])), [1, 3, 2]
    )
    with pytest.raises(ValueError):
        hard_assign(np.array([[np.nan, 1]]))


def test_cvss_precedence_and_conflict(settings):
    base = {
        "version": "3.1",
        "type": "Primary",
        "source": "nvd@nist.gov",
        "vector": "a",
        "base_score": 7.0,
    }
    cna = {**base, "version": "4.0", "source": "cna", "base_score": 8.0}
    versions = settings["cvss_version_order"]
    assert select_cvss(json.dumps([base, cna]), versions)["cvss"] == 8
    assert select_cvss("[]", versions)["cvss"] is None
    secondary = {**base, "source": "other", "base_score": 9.0}
    assert select_cvss(json.dumps([secondary, base]), versions)["cvss"] == 7
    with pytest.raises(ValueError, match="Conflicting"):
        select_cvss(json.dumps([base, {**base, "base_score": 4.0}]), versions)


def inputs(settings):
    settings.update(panel_start="2021-11", panel_end="2021-12", learned_topics=2)
    assignments = pd.DataFrame(
        {
            "cve_id": ["a", "b", "c"],
            "publication_month": ["2021-11", "2021-12", "2021-12"],
            "category": [1, 1, 2],
            "cvss": [0.0, None, 10.0],
        }
    )
    epss = pd.DataFrame({"cve_id": ["c"], "month": ["2021-12"], "epss": [0.0]})
    kev = pd.DataFrame({"cveID": ["a", "older"], "addition_month": ["2021-11", "2021-12"]})
    nvd = pd.DataFrame({"month": ["2021-11", "2021-12"], "feed_coverage_verified": [True, True]})
    coverage = pd.DataFrame(
        {
            "month": ["2021-11", "2021-12"],
            "snapshot_date": ["2021-11-30", "2021-12-31"],
            "snapshot_age_days": [0, 0],
            "snapshot_available": [True, True],
            "model_version": ["v1", "v1"],
            "source_hash": ["h", "h"],
        }
    )
    return assignments, epss, kev, nvd, coverage


def test_panels_conserve_counts_mask_empty_marks_and_keep_kev(settings):
    args = inputs(settings)
    panel, kev = topic_panel(*args, settings)
    assert panel.N.sum() == 3
    assert panel.m_N.all()
    assert panel.loc[panel.N.eq(0), "B"].isna().all()
    assert panel.loc[panel.N.eq(0), "m_B"].eq(False).all()
    dec = panel[panel.month.eq("2021-12")].set_index("category")
    assert dec.loc[1, "B_missing_reason"] == "no_selected_score"
    assert dec.loc[2, "B_coarsened"] == 9.95
    assert dec.loc[2, "m_E"] and dec.loc[2, "E"] == 0
    assert dec.loc[1, "n_E"] == 0 and not dec.loc[1, "m_E"]
    assert panel.loc[panel.month.eq("2021-11"), "V"].isna().all()
    assert dec.loc[3, "V"] == 1 and kev.iloc[1].category == 3


def test_unavailable_counts_are_not_zeros_and_old_snapshot_masked(settings):
    args = inputs(settings)
    args[3].loc[1, "feed_coverage_verified"] = False
    args[4].loc[0, "snapshot_age_days"] = 8
    panel, _ = topic_panel(*args, settings)
    assert panel.loc[panel.month.eq("2021-12"), "N"].isna().all()
    assert not panel.m_E.any()


def test_active_strata_ignore_later_counts(settings):
    panel = pd.DataFrame({"category": [1, 1], "month": ["2018-12", "2024-01"], "N": [1, 999999]})
    assert not active_categories(panel, settings)[0]["active"]


def test_sec_repeat_regular_filing_preserves_multiple_registrants():
    candidates = pd.DataFrame(
        [
            {
                "document_id": "a",
                "item_105": True,
                "filing_retrieved": True,
                "filing_source_hash": "ha",
                "ciks": ["1", "2"],
                "sics": ["6798", "6798"],
                "filing_date": "2024-05-07",
                "accession": "a",
            },
            {
                "document_id": "b",
                "item_105": True,
                "filing_retrieved": True,
                "filing_source_hash": "hb",
                "ciks": ["1", "2"],
                "sics": ["6798", "6798"],
                "filing_date": "2024-06-01",
                "accession": "b",
            },
        ]
    )
    reviews = pd.DataFrame(
        [
            {
                "document_id": "a",
                "disposition": "first_in_sample",
                "event_id": "event",
                "evidence_sha256": "ha",
                "prior_disclosure_requires_check": False,
            },
            {
                "document_id": "b",
                "disposition": "repeat",
                "event_id": "event",
                "evidence_sha256": "hb",
                "prior_disclosure_requires_check": False,
            },
        ]
    )
    bea = pd.DataFrame({"ColCode": ["ORE"], "ColDescr": ["Other real estate"]})
    links, first, panel = incident_tables(
        candidates, reviews, {"6798": "ORE"}, bea, ["2024-05", "2024-06"]
    )
    assert len(links) == 4 and len(first) == 2
    assert panel.reviewed_pairs_first_in_sample.sum() == 2
    assert panel.D.isna().all() and not panel.m_D.any()
    assert panel.eligible_registrants.isna().all()
    bad = deepcopy(reviews)
    bad.loc[1, "evidence_sha256"] = "bad"
    with pytest.raises(ValueError, match="evidence"):
        incident_tables(candidates, bad, {"6798": "ORE"}, bea, ["2024-05"])


def test_count_conservation_failure(settings):
    args = inputs(settings)
    panel, _ = topic_panel(*args, settings)
    panel.loc[0, "N"] += 1
    with pytest.raises(ValueError, match="conservation"):
        validate_panel(panel, args[0], settings)


def test_build_refuses_overwrite_and_path_escape(tmp_path):
    (tmp_path / "data/processed/existing").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        build_panels(tmp_path, Path("source.json"), "existing")
    with pytest.raises(ValueError, match="identifier"):
        build_panels(tmp_path, Path("source.json"), "../escape")
    with pytest.raises(ValueError, match="inside"):
        build_panels(tmp_path, Path("../source.json"), "new")


def test_panel_manifest_rejects_corrupted_upstream_and_escape(tmp_path):
    m = {
        "schema_version": 1,
        "vintage_track": "revised",
        "source_data_manifest": "../escape",
        "source_data_manifest_sha256": "fake",
    }
    path = tmp_path / "panel.json"
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_panels(tmp_path, path)
    m["source_data_manifest"] = "source.json"
    (tmp_path / "source.json").write_text("changed")
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_panels(tmp_path, path)


def test_eligibility_checks_version_at_origin_without_enabling_scores():
    months = pd.period_range("2010-01", "2025-12", freq="M").astype(str)
    panel = pd.DataFrame(
        {
            "month": months,
            "category": 1,
            "m_N": True,
            "m_B": True,
            "m_E": True,
            "m_V": True,
            "B_missing_reason": "observed",
            "E_missing_reason": "observed",
            "V_missing_reason": "observed",
            "epss_snapshot_date": months,
            "epss_model_version": ["v4" if m >= "2025-03" else "v3" for m in months],
        }
    )
    sectors = pd.DataFrame(
        {"month": months, "sector": "unknown", "D_missing_reason": "uncertified"}
    )
    rows = pd.DataFrame(cell_eligibility(panel, sectors, "sourcehash"))
    e = rows[(rows.channel == "E") & (rows.horizon == 1) & (rows.split == "evaluation")]
    assert not e.loc[e.target_month == "2025-03", "target_eligible"].item()
    assert e.loc[e.target_month == "2025-04", "target_eligible"].item()
    assert not rows.eligible_for_scoring.any()
    assert len(rows[(rows.channel == "N") & (rows.split == "evaluation")]) == 78
