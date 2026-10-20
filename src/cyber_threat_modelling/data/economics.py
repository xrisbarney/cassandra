"""BEA source tables and a typed evidence register; no calibration or propagation."""

import json
from urllib.parse import urlencode

from .store import DownloadError


def bea_results(payload):
    results = payload["BEAAPI"]["Results"]
    if isinstance(results, dict) and "Error" in results:
        raise ValueError("BEA returned an API error; inspect the private raw artifact.")
    return results


def ingest_bea(store, output, key):
    import pandas as pd

    if not key:
        return {"status": "unavailable", "reason": "BEA_API_KEY_not_provided"}

    def request(name, **params):
        url = "https://apps.bea.gov/api/data?" + urlencode(
            {"UserID": key, "ResultFormat": "JSON", **params}
        )
        path, record = store.fetch(name, url, interval=0.7)
        # BEA echoes the key in Request. Raw bytes stay local; only Results are normalized.
        store.describe(name, raw_contains_request_credentials=True)
        return bea_results(json.loads(path.read_text(encoding="utf-8"))), record

    table_list, _ = request(
        "bea/tables.json",
        method="GetParameterValues",
        datasetname="InputOutput",
        ParameterName="TableID",
    )
    names = {item["Key"]: item["Desc"] for item in table_list["ParamValue"]}
    tables = []
    for table in ("259", "262", "61"):
        result, record = request(
            f"bea/table-{table}.json",
            method="GetData",
            datasetname="InputOutput",
            TableID=table,
            Year=",".join(str(y) for y in range(2010, 2026)),
        )
        blocks = result if isinstance(result, list) else [result]
        rows = [row for block in blocks for row in block.get("Data", [])]
        if not rows:
            raise ValueError("Empty BEA table.")
        frame = pd.DataFrame(rows)
        if frame.duplicated(["TableID", "Year", "RowCode", "ColCode"]).any():
            raise ValueError("Duplicate BEA table coordinates.")
        frame["source_hash"] = record["sha256"]
        frame.to_parquet(output / f"bea_{table}.parquet", index=False)
        tables.append(
            {
                "table_id": table,
                "description": names[table],
                "rows": len(frame),
                "years": sorted(frame.Year.unique().tolist()),
                "units": "coefficients" if table == "61" else "millions_of_current_dollars",
                "units_status": "table_family_convention_requires_Stage11_confirmation",
                "geography": "United States",
                "historical_vintages_verified": False,
                "source_hash": record["sha256"],
            }
        )
    return {
        "status": "retrieved",
        "tables": tables,
        "economic_readiness": (
            "requires_classification_units_and_shock_interpretation_checks_Stage11"
        ),
    }


def validate_event_evidence(rows):
    """Reject undocumented amounts and prevent scenario ranges becoming observed losses."""
    required = {
        "event_id",
        "source_url",
        "publication_date",
        "geography",
        "loss_definition",
        "currency",
        "price_year",
        "period_start",
        "period_end",
        "amount_lower",
        "amount_upper",
        "evidence_type",
        "calibration_eligible",
        "exclusion_reason",
    }
    for row in rows:
        if set(row) != required:
            raise ValueError("Event evidence must use the documented schema.")
        if row["amount_lower"] is not None or row["amount_upper"] is not None:
            if row["amount_lower"] is None or row["amount_upper"] is None:
                raise ValueError("Both amount endpoints are required.")
            if not 0 <= row["amount_lower"] <= row["amount_upper"]:
                raise ValueError("Invalid event-loss range.")
        if row["calibration_eligible"] and (
            row["evidence_type"] != "documented_realised_loss"
            or row["amount_lower"] is None
            or any(
                row[k] in (None, "unknown")
                for k in (
                    "geography",
                    "loss_definition",
                    "price_year",
                    "period_start",
                    "period_end",
                )
            )
        ):
            raise ValueError("Incompatible or incomplete loss evidence cannot calibrate damage.")
    return rows


def ingest_events(store, output, register_path):
    from .store import write_json

    rows = validate_event_evidence(json.loads(register_path.read_text(encoding="utf-8")))
    sources = []
    for row in rows:
        name = f"events/{row['event_id']}.source"
        try:
            _, record = store.fetch(name, row["source_url"])
            store.describe(
                name,
                source_date=row["publication_date"],
                publication_date=row["publication_date"],
                source_version="issuer_or_government_document",
                exclusions=[row["exclusion_reason"]],
            )
            sources.append(
                {"event_id": row["event_id"], "retrieved": True, "source_hash": record["sha256"]}
            )
        except DownloadError:
            sources.append({"event_id": row["event_id"], "retrieved": False})
    write_json(output / "event_evidence.json", rows)
    return {
        "sources": sources,
        "calibration_eligible": sum(r["calibration_eligible"] for r in rows),
        "economic_claim": "no_compatible_national_monthly_loss_target_established",
    }
