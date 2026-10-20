"""Resumable raw downloads with byte hashes and credential-free request metadata."""

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def credentials(path=None):
    """Read only explicitly supported names; never log values or modify the environment."""
    allowed = {"NVD_API_KEY", "BEA_API_KEY", "SEC_USER_AGENT"}
    result = {key: os.environ[key] for key in allowed if os.environ.get(key)}
    if path:
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
            key, separator, value = line.strip().partition("=")
            if separator and key in allowed and key not in result:
                result[key] = value.strip().strip("\"'")
    return result


def public_url(url):
    parts = urllib.parse.urlsplit(url)
    sensitive = {"userid", "apikey", "api_key", "token", "key"}
    query = [
        (k, "REDACTED" if k.lower() in sensitive else v)
        for k, v in urllib.parse.parse_qsl(parts.query)
    ]
    return urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urllib.parse.urlencode(sorted(query)), "")
    )


class DownloadError(RuntimeError):
    pass


class RawStore:
    def __init__(
        self, root, user_agent="Cassandra research; https://github.com/xrisbarney/cassandra"
    ):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.user_agent = user_agent
        self.records = {}
        self.last_request = {}

    def fetch(self, name, url, *, headers=None, interval=0.3):
        destination = (self.root / name).resolve()
        if not destination.is_relative_to(self.root):
            raise ValueError("Raw file path escapes the snapshot.")
        sidecar = destination.with_name(destination.name + ".record.json")
        if sidecar.exists():
            record = json.loads(sidecar.read_text(encoding="utf-8"))
            if (
                public_url(record["url"]) != public_url(url)
                or sha256(destination) != record["sha256"]
            ):
                raise ValueError("Cached raw artifact does not match its request/hash.")
            self.records[name] = record
            return destination, record
        destination.parent.mkdir(parents=True, exist_ok=True)
        host = urllib.parse.urlsplit(url).netloc
        for attempt in range(3):
            time.sleep(max(0, interval - (time.monotonic() - self.last_request.get(host, 0))))
            self.last_request[host] = time.monotonic()
            try:
                request = urllib.request.Request(
                    url, headers={"User-Agent": self.user_agent, **(headers or {})}
                )
                with urllib.request.urlopen(request, timeout=60) as response:
                    partial = destination.with_name(destination.name + ".part")
                    with partial.open("wb") as output:
                        while chunk := response.read(1024 * 1024):
                            output.write(chunk)
                    expected = response.headers.get("Content-Length")
                    if expected and partial.stat().st_size != int(expected):
                        raise DownloadError("Truncated HTTP response.")
                    partial.replace(destination)
                    record = {
                        "source_id": name,
                        "url": public_url(url),
                        "retrieved_at": utc_now(),
                        "http_status": response.status,
                        "content_type": response.headers.get("Content-Type"),
                        "http_last_modified": response.headers.get("Last-Modified"),
                        "http_etag": response.headers.get("ETag"),
                        "sha256": sha256(destination),
                        "bytes": destination.stat().st_size,
                        "available_at": "unknown",
                        "source_date": "unknown",
                        "source_version": "unknown",
                        "revision_policy": "pinned_retrieval",
                        "exclusions": [],
                    }
                    write_json(sidecar, record)
                    self.records[name] = record
                    return destination, record
            except urllib.error.HTTPError as error:
                status = error.code
                body = error.read()
                if status not in (429, 500, 502, 503, 504) or attempt == 2:
                    failure = destination.with_name(destination.name + ".http-error")
                    failure.write_bytes(body)
                    record = {
                        "source_id": name,
                        "url": public_url(url),
                        "http_status": status,
                        "retrieved_at": utc_now(),
                        "sha256": sha256(failure),
                        "bytes": len(body),
                        "available_at": "unknown",
                        "source_date": "unknown",
                        "source_version": "unknown",
                        "revision_policy": "failed_request",
                        "exclusions": ["http_error"],
                    }
                    self.records[name] = record
                    write_json(failure.with_name(failure.name + ".record.json"), record)
                    raise DownloadError(f"HTTP {status} for source {name}") from None
            except (OSError, DownloadError):
                if attempt == 2:
                    raise DownloadError(f"Download failed for source {name}") from None
            time.sleep(2**attempt)
        raise DownloadError(f"Download failed for source {name}")

    def describe(self, name, **fields):
        # Enriched manifest metadata is separate from immutable raw-response sidecars.
        self.records[name].update(fields)
