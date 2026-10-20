"""Observable monthly targets; preserve unavailable observations as nulls."""

import json

import numpy as np
import pandas as pd


def select_cvss(payload, versions):
    entries = json.loads(payload)
    if not entries:
        return {"cvss": None, "cvss_version": None, "cvss_source": None, "cvss_vector": None}

    def key(entry):
        if entry["version"] not in versions:
            raise ValueError("Unrecognized CVSS version; review the score policy.")
        primary = entry["type"] == "Primary"
        nvd = entry["source"] == "nvd@nist.gov"
        rank = 0 if primary and nvd else 1 if primary else 2 if nvd else 3
        return (versions.index(entry["version"]), rank, entry["source"], entry["vector"])

    ordered = sorted(entries, key=key)
    winner = ordered[0]
    tied = [e for e in ordered if key(e) == key(winner)]
    if len({e["base_score"] for e in tied}) != 1:
        raise ValueError("Conflicting equally ranked CVSS entries require review.")
    if not all(np.isfinite(e["base_score"]) and 0 <= e["base_score"] <= 10 for e in entries):
        raise ValueError("CVSS score outside [0,10].")
    return {
        "cvss": winner["base_score"],
        "cvss_version": winner["version"],
        "cvss_source": winner["source"],
        "cvss_vector": winner["vector"],
    }


def topic_panel(assignments, epss, kev, nvd_coverage, epss_coverage, settings):
    if assignments.cve_id.duplicated().any() or epss.cve_id.duplicated().any():
        raise ValueError("Duplicate CVE in publication cohort or EPSS join.")
    months = pd.period_range(settings["panel_start"], settings["panel_end"], freq="M").astype(str)
    categories = range(1, settings["learned_topics"] + 2)
    index = pd.MultiIndex.from_product([months, categories], names=["month", "category"])
    cohort = assignments.rename(columns={"publication_month": "month"})
    grouped = cohort.groupby(["month", "category"])
    panel = pd.DataFrame(index=index)
    panel["N"] = grouped.size().reindex(index, fill_value=0).astype("Int64")
    panel["n_B"] = grouped.cvss.count().reindex(index, fill_value=0).astype(int)
    panel["B"] = grouped.cvss.mean().reindex(index)
    nvd_flags = nvd_coverage.set_index("month").feed_coverage_verified
    if not set(months).issubset(nvd_flags.index):
        raise ValueError("NVD coverage must explicitly cover every panel month.")
    panel["m_N"] = [bool(nvd_flags[m]) for m, _ in index]
    panel.loc[~panel.m_N, "N"] = pd.NA
    panel["m_B"] = panel.m_N & (panel.n_B > 0)
    panel.loc[~panel.m_B, "B"] = np.nan
    panel["B_coarsened"] = panel.B.clip(*settings["severity_clip"])
    panel["B_missing_reason"] = np.select(
        [~panel.m_N, panel.N.eq(0).fillna(False), panel.n_B.eq(0)],
        ["cohort_unavailable", "empty_cohort", "no_selected_score"],
        default="observed",
    )
    joined = epss.merge(
        cohort[["cve_id", "month", "category"]],
        on=["cve_id", "month"],
        how="left",
        validate="one_to_one",
    )
    if joined.category.isna().any():
        raise ValueError("EPSS records do not match their publication-month cohort.")
    if not joined.epss.between(0, 1).all():
        raise ValueError("Invalid EPSS probability.")
    egroup = joined.groupby(["month", "category"])
    panel["n_E"] = egroup.epss.count().reindex(index, fill_value=0).astype(int)
    panel["E"] = egroup.epss.mean().reindex(index)
    coverage = epss_coverage.set_index("month")
    for column in ("snapshot_date", "snapshot_age_days", "model_version", "source_hash"):
        panel["epss_" + column] = [coverage[column].get(m, None) for m, _ in index]
    available = coverage.snapshot_available
    snapshot_ok = np.array([bool(available.get(m, False)) for m, _ in index])
    age = pd.to_numeric(panel.epss_snapshot_age_days)
    snapshot_ok &= age.between(0, 7).to_numpy()
    dates = pd.to_datetime(panel.epss_snapshot_date)
    ends = pd.to_datetime([pd.Period(m, freq="M").end_time for m, _ in index])
    snapshot_ok &= np.asarray(dates <= ends)
    panel["m_E"] = panel.m_N & snapshot_ok & (panel.n_E > 0)
    panel.loc[~panel.m_E, "E"] = np.nan
    panel["E_coarsened"] = panel.E.clip(*settings["epss_clip"])
    panel["E_missing_reason"] = np.select(
        [~panel.m_N, ~snapshot_ok, panel.N.eq(0).fillna(False), panel.n_E.eq(0)],
        ["cohort_unavailable", "snapshot_unavailable", "empty_cohort", "no_cohort_match"],
        default="observed",
    )
    if (panel.n_B > panel.N).any() or (panel.n_E > panel.N).any():
        raise ValueError("More marks than CVEs in a cohort.")
    additions = kev[kev.addition_month.between(months[0], months[-1])].copy()
    if additions.cveID.duplicated().any():
        raise ValueError("Duplicate KEV addition.")
    lookup = assignments.set_index("cve_id").category
    additions["category"] = additions.cveID.map(lookup).fillna(categories[-1]).astype(int)
    additions["topic_source"] = np.where(
        additions.cveID.isin(lookup.index),
        "frozen_nvd_assignment",
        "unknown_outside_retained_cohort",
    )
    v = additions.groupby(["addition_month", "category"]).size()
    panel["V_catalogued"] = v.reindex(index, fill_value=0).astype(int)
    panel["m_V"] = [m >= "2021-12" for m, _ in index]
    panel["V"] = panel.V_catalogued.astype("Int64").where(panel.m_V)
    panel["V_missing_reason"] = [
        "observed" if m >= "2021-12" else "launch_backlog" if m == "2021-11" else "pre_catalogue"
        for m, _ in index
    ]
    panel["vintage_track"] = "revised"
    panel["metadata_available_at"] = "unknown"
    return panel.reset_index(), additions


def active_categories(panel, settings):
    initial = panel[panel.month.between(settings["training_start"], settings["training_end"])]
    rows = []
    for category, block in initial.groupby("category"):
        total, nonzero = int(block.N.sum()), int((block.N > 0).sum())
        rows.append(
            {
                "category": int(category),
                "initial_assignments": total,
                "initial_nonzero_months": nonzero,
                "active": total >= settings["active_min_assignments"]
                and nonzero >= settings["active_min_nonzero_months"],
            }
        )
    return rows


def cell_eligibility(panel, sectors, source_manifest_hash):
    """Target availability is distinct from fit readiness and score execution permission."""
    topics = {m: block for m, block in panel.groupby("month")}
    sector_blocks = {m: block for m, block in sectors.groupby("month")}
    for split, start, end in (
        ("development", "2019-01", "2023-12"),
        ("evaluation", "2024-01", "2025-12"),
    ):
        first, last = pd.Period(start, freq="M"), pd.Period(end, freq="M")
        for h in (1, 3, 6, 12):
            for origin in pd.period_range(first - 1, last - h, freq="M"):
                target = str(origin + h)
                for row in topics[target].itertuples():
                    for channel in ("N", "B", "E", "V"):
                        observed = bool(getattr(row, "m_" + channel))
                        # Conservative default: a new EPSS version needs observed training
                        # support. Future announcement handling is a later origin-aware task.
                        history = panel[
                            (panel.month <= str(origin)) & (panel.category == row.category)
                        ]
                        supported = channel != "E" or bool(
                            (
                                history.m_E & history.epss_model_version.eq(row.epss_model_version)
                            ).any()
                        )
                        reason = (
                            "observed"
                            if observed
                            else "source_unavailable"
                            if channel == "N"
                            else getattr(row, channel + "_missing_reason")
                        )
                        if observed and not supported:
                            reason = "epss_version_without_observed_training_support"
                        yield {
                            "split": split,
                            "origin_month": str(origin),
                            "target_month": target,
                            "horizon": h,
                            "channel": channel,
                            "category": str(row.category),
                            "vintage_track": "revised",
                            "source_date": row.epss_snapshot_date if channel == "E" else target,
                            "available_at": "unknown",
                            "source_manifest_sha256": source_manifest_hash,
                            "observed_flag": observed,
                            "target_eligible": observed and supported,
                            "eligible_for_scoring": False,
                            "eligibility_reason": reason,
                            "execution_gate": "stage6_pre_scoring_checks_pending",
                        }
                for row in sector_blocks[target].itertuples():
                    yield {
                        "split": split,
                        "origin_month": str(origin),
                        "target_month": target,
                        "horizon": h,
                        "channel": "D",
                        "category": row.sector,
                        "vintage_track": "revised",
                        "source_date": target,
                        "available_at": "unknown",
                        "source_manifest_sha256": source_manifest_hash,
                        "observed_flag": False,
                        "target_eligible": False,
                        "eligible_for_scoring": False,
                        "eligibility_reason": row.D_missing_reason,
                        "execution_gate": "SEC_coverage_not_certified",
                    }
