"""Targeted historical-vintage audit; samples cannot certify an entire as-of panel."""

import json
from urllib.parse import urlencode

from .store import DownloadError


def audit_history(store, key=None):
    headers = {"apiKey": key} if key else {}
    interval = 0.7 if key else 6.1
    samples = []
    for identifier in ("CVE-2010-3333", "CVE-2021-44228", "CVE-2024-3400"):
        name = f"nvd-history/{identifier}.json"
        try:
            path, record = store.fetch(
                name,
                "https://services.nvd.nist.gov/rest/json/cvehistory/2.0?"
                + urlencode({"cveId": identifier}),
                headers=headers,
                interval=interval,
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            changes = payload.get("cveChanges", [])
            events = [item.get("change", item) for item in changes]
            samples.append(
                {
                    "cve_id": identifier,
                    "events_returned": len(events),
                    "total_results": payload.get("totalResults"),
                    "event_names": sorted({event.get("eventName", "unknown") for event in events}),
                    "detail_types": sorted(
                        {
                            detail.get("type", "unknown")
                            for event in events
                            for detail in event.get("details", [])
                        }
                    ),
                    "source_hash": record["sha256"],
                }
            )
        except DownloadError:
            samples.append({"cve_id": identifier, "status": "unavailable"})
    return {
        "samples": samples,
        "point_in_time_panel_certified": False,
        "reason": (
            "event_logs_do_not_by_themselves_certify_full_field_snapshots_for_every_CVE_and_cutoff"
        ),
    }
