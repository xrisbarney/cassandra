import gzip
import hashlib
import io
import json
from datetime import date

import pytest

from cyber_threat_modelling.data.economics import bea_results, validate_event_evidence
from cyber_threat_modelling.data.nvd import normalize_cve, validate_feed
from cyber_threat_modelling.data.sec import sec_candidate
from cyber_threat_modelling.data.signals import parse_kev, read_epss, select_snapshot
from cyber_threat_modelling.data.store import RawStore, credentials, public_url


def test_raw_resume_hash_verification_and_redacted_request(tmp_path, monkeypatch):
    class Response(io.BytesIO):
        status = 200
        headers = {"Content-Length": "7", "Content-Type": "application/json"}

    calls = []

    def open_url(request, **kwargs):
        calls.append(request.full_url)
        return Response(b'{"x":1}')

    monkeypatch.setattr("urllib.request.urlopen", open_url)
    store = RawStore(tmp_path)
    url = "https://example.org/data?UserID=private-value&method=GetData"
    path, record = store.fetch("bea/data.json", url, interval=0)
    assert path.read_bytes() == b'{"x":1}'
    assert "private-value" not in json.dumps(record)
    store.fetch("bea/data.json", url, interval=0)
    assert len(calls) == 1
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        store.fetch("bea/data.json", url, interval=0)


def test_raw_paths_and_query_canonicalization(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        RawStore(tmp_path).fetch("../outside", "https://example.org")
    assert public_url("https://example.org/?b=2&a=1") == public_url("https://example.org/?a=1&b=2")


def test_credentials_only_load_supported_names(tmp_path, monkeypatch):
    for key in ("NVD_API_KEY", "BEA_API_KEY", "SEC_USER_AGENT"):
        monkeypatch.delenv(key, raising=False)
    path = tmp_path / ".env"
    path.write_text('NVD_API_KEY="example"\nUNRELATED_SECRET=exclude\n')
    assert credentials(path) == {"NVD_API_KEY": "example"}


def test_nvd_checksum_is_for_uncompressed_bytes(tmp_path):
    payload = b'{"vulnerabilities":[]}'
    path = tmp_path / "feed.gz"
    path.write_bytes(gzip.compress(payload))
    meta = {"size": str(len(payload)), "sha256": hashlib.sha256(payload).hexdigest().upper()}
    validate_feed(path, meta)
    meta["size"] = "1"
    with pytest.raises(ValueError, match="checksum"):
        validate_feed(path, meta)


def test_cve_uses_publication_month_and_preserves_missing_marks():
    cve = {
        "id": "CVE-2009-1234",
        "published": "2010-02-01T00:00:00.000",
        "lastModified": "2026-01-01T00:00:00.000",
        "vulnStatus": "Rejected",
    }
    result = normalize_cve(cve, "source", "hash")
    assert result["publication_month"] == "2010-02"
    assert result["rejected"]
    assert result["cvss_json"] == "[]" and result["description_en"] == ""
    assert result["metadata_available_at"] == "unknown"


@pytest.mark.parametrize(
    "header", ["cve,epss\n", "#model_version:v3,score_date:2024-01-31\ncve,epss,percentile\n"]
)
def test_epss_old_and_new_schema_preserve_zero(tmp_path, header):
    path = tmp_path / "epss.gz"
    line = "CVE-2020-0001,0,0.1\n" if "percentile" in header else "CVE-2020-0001,0\n"
    path.write_bytes(gzip.compress((header + line).encode()))
    _, scores = read_epss(path)
    assert scores == {"CVE-2020-0001": 0.0}
    assert "CVE-2020-0002" not in scores


@pytest.mark.parametrize("value", ["nan", "1.1", "-0.1"])
def test_epss_invalid_probabilities_rejected(tmp_path, value):
    path = tmp_path / "epss.gz"
    path.write_bytes(gzip.compress(f"cve,epss\nCVE-2020-0001,{value}\n".encode()))
    with pytest.raises(ValueError):
        read_epss(path)


def test_epss_cutoff_excludes_future_and_stale_snapshots():
    end = date(2024, 1, 31)
    assert select_snapshot(end, [date(2024, 2, 1), date(2024, 1, 23)]) is None
    assert select_snapshot(end, [date(2024, 1, 24)]) == date(2024, 1, 24)


def test_kev_checks_duplicates_and_declared_count():
    item = {"cveID": "CVE-2021-0001", "dateAdded": "2021-11-03"}
    assert parse_kev({"count": 1, "vulnerabilities": [item]}) == [item]
    with pytest.raises(ValueError, match="count"):
        parse_kev({"count": 2, "vulnerabilities": [item]})
    with pytest.raises(ValueError, match="Duplicate"):
        parse_kev({"count": 2, "vulnerabilities": [item, item]})


def test_sec_amendment_and_multi_registrant_are_candidates_not_incidents():
    hit = {
        "_id": "accession:document",
        "_source": {
            "adsh": "accession",
            "file_date": "2024-01-01",
            "form": "8-K/A",
            "ciks": ["1", "2"],
            "items": ["1.05"],
        },
    }
    row = sec_candidate(hit)
    assert row["amendment"] and row["item_105"] and row["ciks"] == ["1", "2"]
    assert row["incident_id"] is None and not row["observed_incident"]


def test_bea_api_error_not_silently_treated_as_data():
    with pytest.raises(ValueError, match="API error"):
        bea_results({"BEAAPI": {"Results": {"Error": {"APIErrorCode": "1"}}}})


def test_event_scope_mismatch_cannot_be_calibration_evidence():
    from pathlib import Path

    path = Path(__file__).parents[1] / "configs" / "event_evidence.json"
    rows = json.loads(path.read_text())
    assert len(validate_event_evidence(rows)) == 4
    rows[-1]["calibration_eligible"] = True
    with pytest.raises(ValueError, match="cannot calibrate"):
        validate_event_evidence(rows)


def sample_manifest(root):
    from cyber_threat_modelling.data.store import sha256, write_json

    raw = root / "data" / "raw" / "tiny.json"
    write_json(raw, {"value": 1})
    source = root / "source_manifest.json"
    write_json(source, {"records": [{"path": "data/raw/tiny.json", "sha256": sha256(raw)}]})
    policies = {
        "policy_sha256": "docs/DATA_DICTIONARY.md",
        "event_register_sha256": "configs/event_evidence.json",
        "protocol_freeze_sha256": "docs/STAGE2_FREEZE.json",
    }
    manifest = {
        "schema_version": 1,
        "vintage_track": "revised",
        "snapshot": "tiny",
        "scoring_ready": False,
        "source_manifest": "source_manifest.json",
        "source_manifest_sha256": sha256(source),
        "artifacts": {},
    }
    for key, relative in policies.items():
        path = root / relative
        if not path.exists():
            write_json(path, {})
        manifest[key] = sha256(path)
    destination = root / "data_manifest.json"
    write_json(destination, manifest)
    return destination, manifest


def test_data_manifest_rejects_wrong_vintage_and_tampering(tmp_path):
    from cyber_threat_modelling.data.audit import verify_data_manifest

    path, _ = sample_manifest(tmp_path)
    assert not verify_data_manifest(tmp_path, path)["scoring_ready"]
    with pytest.raises(ValueError, match="vintage"):
        verify_data_manifest(tmp_path, path, "point_in_time")
    (tmp_path / "data/raw/tiny.json").write_text("changed")
    with pytest.raises(ValueError, match="hash"):
        verify_data_manifest(tmp_path, path)


def test_data_manifest_rejects_escaping_artifact_path(tmp_path):
    from cyber_threat_modelling.data.audit import verify_data_manifest
    from cyber_threat_modelling.data.store import write_json

    path, manifest = sample_manifest(tmp_path)
    manifest["artifacts"] = {"../outside": "any"}
    write_json(path, manifest)
    with pytest.raises(ValueError, match="escapes"):
        verify_data_manifest(tmp_path, path)


def test_prepare_run_attaches_verified_data_and_refuses_corruption(project):
    from cyber_threat_modelling.config import load_config
    from cyber_threat_modelling.provenance import prepare_run

    path, _ = sample_manifest(project)
    config = load_config(project / "configs/pilot.toml")
    record = prepare_run(project, config, project / "run", path)
    assert json.loads(record.read_text())["data_manifest"]["snapshot"] == "tiny"
    (project / "data/raw/tiny.json").write_text("changed")
    with pytest.raises(ValueError, match="hash"):
        prepare_run(project, config, project / "invalid", path)
    assert not (project / "invalid").exists()


def test_frozen_audit_cannot_be_overwritten(tmp_path):
    from cyber_threat_modelling.data.audit import finalize_audit, run_audit
    from cyber_threat_modelling.data.store import write_json

    path = tmp_path / "docs/audits/frozen/data_manifest.json"
    write_json(path, {})
    for operation in (run_audit, finalize_audit):
        with pytest.raises(FileExistsError, match="frozen"):
            operation(tmp_path, "frozen")
    assert path.read_text().strip() == "{}"


def test_source_eligibility_is_not_forecast_permission():
    import pandas as pd

    from cyber_threat_modelling.data.audit import eligibility_rows

    monthly = [
        {
            "month": str(m),
            "feed_coverage_verified": True,
            "nonrejected_records": 1,
            "missing_cvss": 0,
        }
        for m in pd.period_range("2010-01", "2025-12", freq="M")
    ]
    frame = pd.DataFrame(eligibility_rows({"nvd": {"monthly": monthly}, "epss": {"months": []}}))
    assert len(frame) == 1500
    assert not frame.eligible_for_scoring.any() and not frame.observed_flag.any()
    assert not frame.loc[frame.channel.isin(["D", "E"]), "source_available"].any()
    assert not frame.loc[
        (frame.channel == "V") & (frame.target_month <= "2021-11"), "source_available"
    ].any()
    assert (frame.origin_month < frame.target_month).all()
    assert frame[(frame.split == "evaluation") & (frame.channel == "N")].groupby(
        "horizon"
    ).size().to_dict() == {1: 24, 3: 22, 6: 19, 12: 13}
