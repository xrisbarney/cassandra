"""Reviewed SEC records and a masked sector panel; search hits never imply coverage."""

import pandas as pd


def incident_tables(candidates, reviews, sic_map, bea, months):
    if reviews.document_id.duplicated().any():
        raise ValueError("Duplicate filing adjudication.")
    if set(candidates.document_id) != set(reviews.document_id):
        raise ValueError("Every SEC candidate needs an explicit disposition.")
    merged = candidates.merge(
        reviews, on="document_id", validate="one_to_one", suffixes=("", "_review")
    )
    pairs = []
    for row in merged.itertuples():
        if not row.item_105:
            if row.disposition != "excluded_not_item105":
                raise ValueError("Non-Item-1.05 record incorrectly included.")
            continue
        if not row.filing_retrieved or row.evidence_sha256 != row.filing_source_hash:
            raise ValueError("SEC review evidence does not match the retrieved filing.")
        if row.disposition not in ("first_in_sample", "repeat") or not row.event_id:
            raise ValueError("Unresolved SEC incident linkage.")
        for i, cik in enumerate(row.ciks):
            # SEC array lengths need not match. Never guess a missing registrant's SIC.
            sic = str(row.sics[i]) if len(row.sics) == len(row.ciks) else None
            sector = sic_map.get(sic, "unknown")
            pairs.append(
                {
                    "event_id": row.event_id,
                    "cik": str(cik),
                    "sic": sic,
                    "sector": sector,
                    "filing_date": row.filing_date,
                    "document_id": row.document_id,
                    "accession": row.accession,
                    "disposition": row.disposition,
                    "source_hash": row.filing_source_hash,
                    "prior_disclosure_requires_check": row.prior_disclosure_requires_check,
                }
            )
    links = pd.DataFrame(pairs).sort_values(["filing_date", "document_id", "cik"])
    for _, block in links.groupby(["event_id", "cik"]):
        if (
            block.iloc[0].disposition != "first_in_sample"
            or (block.disposition == "first_in_sample").sum() != 1
        ):
            raise ValueError(
                "Each incident-registrant pair needs exactly one first-in-sample filing."
            )
        if block.sector.nunique() != 1:
            raise ValueError("Sector changed within an incident; explicit review required.")
    first = links.drop_duplicates(["event_id", "cik"]).copy()
    first["month"] = first.filing_date.str[:7]
    first["first_qualifying_disclosure_verified"] = False
    labels = (
        bea[["ColCode", "ColDescr"]]
        .drop_duplicates()
        .rename(columns={"ColCode": "sector", "ColDescr": "sector_name"})
    )
    if labels.sector.duplicated().any():
        raise ValueError("BEA sector labels conflict.")
    labels = pd.concat([labels, pd.DataFrame([{"sector": "unknown", "sector_name": "Unmapped"}])])
    if not set(first.sector).issubset(set(labels.sector)):
        raise ValueError("SEC crosswalk contains a sector absent from the BEA source.")
    index = pd.MultiIndex.from_product([months, labels.sector], names=["month", "sector"])
    panel = pd.DataFrame(index=index)
    panel["reviewed_pairs_first_in_sample"] = (
        first.groupby(["month", "sector"]).size().reindex(index, fill_value=0)
    )
    panel["D"] = pd.array([pd.NA] * len(panel), dtype="Int64")
    panel["m_D"] = False
    panel["eligible_registrants"] = float("nan")
    panel["q_reporting_step"] = [int(m >= "2024-07") for m, _ in index]
    panel["D_missing_reason"] = [
        "pre_mandate"
        if m < "2023-12"
        else "partial_compliance_month"
        if m in ("2023-12", "2024-06")
        else "search_recall_and_population_uncertified"
        for m, _ in index
    ]
    return links, first, panel.reset_index().merge(labels, on="sector", validate="many_to_one")
