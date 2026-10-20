"""Install a checksum-pinned Windows compiler and add its bin directory to user PATH.

Uses only the standard library. No administrator rights or Python package changes.
"""

import ctypes
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    if sys.platform != "win32":
        raise SystemExit("This installer is for Windows only.")
    import winreg

    root = Path(__file__).resolve().parents[1]
    spec = json.loads((root / "configs/windows_compiler.json").read_text(encoding="utf-8"))
    cache = root / ".tools/compiler"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "winlibs.zip"
    if not archive.exists() or digest(archive) != spec["sha256"]:
        partial = cache / "winlibs.zip.part"
        print("Downloading pinned WinLibs compiler...", flush=True)
        with urllib.request.urlopen(spec["url"], timeout=60) as response, partial.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        if digest(partial) != spec["sha256"]:
            raise RuntimeError("Compiler archive checksum mismatch; refusing extraction.")
        partial.replace(archive)
    target = (
        Path(os.environ["LOCALAPPDATA"]) / "Programs/CassandraToolchain" / spec["release"]
    ).resolve()
    marker = target / "installation.json"
    if not marker.exists():
        print("Checksum verified. Extracting compiler...", flush=True)
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                if not (target / member.filename).resolve().is_relative_to(target):
                    raise RuntimeError("Archive member escapes installation directory.")
            bundle.extractall(target)
        marker.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    elif json.loads(marker.read_text(encoding="utf-8")) != spec:
        raise RuntimeError("Existing installation differs from the compiler specification.")

    compiler_bin = target / "mingw64/bin"
    compiler = compiler_bin / "g++.exe"
    version = subprocess.check_output([str(compiler), "--version"], text=True).splitlines()[0]
    with winreg.OpenKey(
        winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_READ | winreg.KEY_WRITE
    ) as key:
        try:
            previous, value_type = winreg.QueryValueEx(key, "Path")
        except FileNotFoundError:
            previous, value_type = "", winreg.REG_EXPAND_SZ
        entries = [p for p in previous.split(";") if p]
        backup = cache / "user-path-before.json"
        if not backup.exists():
            backup.write_text(json.dumps({"value": previous, "type": value_type}), encoding="utf-8")
        if str(compiler_bin).lower() not in [p.lower().rstrip("\\/") for p in entries]:
            winreg.SetValueEx(key, "Path", 0, value_type, ";".join([str(compiler_bin), *entries]))
    # Notify Explorer so newly launched terminals can inherit the updated user PATH.
    result = ctypes.c_size_t()
    ctypes.windll.user32.SendMessageTimeoutW(
        65535, 0x001A, 0, "Environment", 2, 5000, ctypes.byref(result)
    )
    record = {
        **spec,
        "compiler_path": str(compiler),
        "compiler_bin": str(compiler_bin),
        "compiler_binary_sha256": digest(compiler),
        "version_output": version,
    }
    (cache / "installation.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    print("Open a new terminal, or dot-source scripts/Use-Compiler.ps1 in an existing terminal.")


if __name__ == "__main__":
    main()
