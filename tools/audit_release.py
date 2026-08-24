#!/usr/bin/env python3
"""Fail when the public release candidate contains likely private material."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_RELEASE_BYTES = 10 * 1024 * 1024

# These two small binary assets were inspected explicitly. Pinning their hashes
# makes any later replacement a deliberate, reviewable change.
PINNED_ASSETS = {
    Path("examples/factory_smoke/frame.npz"):
        "c9d42892f5a23413c7705b28b8460e6814886efcdbdbeba478842939f1868555",
    Path("src/tbfe/preprocess/assets/Precomputed.exr"):
        "aafbd6c78fe7974159af02c7cc9be7999949457bacff8352aa984f0db3fa761a",
}

FORBIDDEN_NAMES = {
    ".env",
    ".gitmodules",
    "download.py",
    "id_rsa",
    "id_ed25519",
}
FORBIDDEN_DIRECTORY_NAMES = {
    ".ssh",
    "checkpoints",
    "data",
    "datasets",
    "outputs",
    "pre_release_weights",
    "result",
    "runs",
    "save",
    "wandb",
    "weights",
}
FORBIDDEN_SUFFIXES = {
    ".7z",
    ".arrow",
    ".bin",
    ".bz2",
    ".ckpt",
    ".data",
    ".engine",
    ".exr",
    ".gz",
    ".h5",
    ".hdf5",
    ".joblib",
    ".key",
    ".npy",
    ".npz",
    ".onnx",
    ".p12",
    ".parquet",
    ".pb",
    ".pem",
    ".pfx",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".rar",
    ".safetensors",
    ".tar",
    ".tflite",
    ".trt",
    ".whl",
    ".xz",
    ".zip",
}

# Construct sensitive literal fragments in pieces so this guard does not flag
# its own source code.
PRIVATE_PATH_FRAGMENTS = (
    b"/" + b"disk/",
    b"/" + b"home/",
    b"/" + b"mnt/",
)
PRIVATE_KEY_MARKER = b"-----BEGIN " + b"PRIVATE KEY-----"
PRIVATE_KEY_VARIANTS = re.compile(
    b"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?" + b"PRIVATE KEY-----"
)
SSH_TRANSFER_CODE = re.compile(
    rb"\b(?:" + b"para" + b"miko|" + b"py" + b"sftp|" + b"ssh" + b"pass" + rb")\b",
    re.IGNORECASE,
)

CONTENT_RULES = (
    ("private-key header", PRIVATE_KEY_VARIANTS),
    ("GitHub token", re.compile(rb"\bgh[pousr]_[A-Za-z0-9_]{20,}\b")),
    ("AWS access key", re.compile(rb"\bAKIA[0-9A-Z]{16}\b")),
    ("Google API key", re.compile(rb"\bAIza[0-9A-Za-z_-]{30,}\b")),
    ("Slack token", re.compile(rb"\bxox[baprs]-[0-9A-Za-z-]{20,}\b")),
    ("Hugging Face token", re.compile(rb"\bhf_[A-Za-z0-9]{20,}\b")),
    ("OpenAI-style token", re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    (
        "credential embedded in URL",
        re.compile(rb"[a-z][a-z0-9+.-]*://[^\s/:@]+:[^\s/@]+@", re.IGNORECASE),
    ),
    (
        "hard-coded secret assignment",
        re.compile(
            rb"\b(?:password|passwd|secret(?:_key)?|access(?:_key)?|auth(?:_token)?)"
            rb"\s*[:=]\s*['\"][^'\"\r\n]{4,}['\"]",
            re.IGNORECASE,
        ),
    ),
    (
        "private-network IPv4 address",
        re.compile(
            rb"(?<![0-9])(?:10\.(?:[0-9]{1,3}\.){2}[0-9]{1,3}|"
            rb"192\.168\.(?:[0-9]{1,3}\.)[0-9]{1,3}|"
            rb"172\.(?:1[6-9]|2[0-9]|3[01])\.(?:[0-9]{1,3}\.)[0-9]{1,3})(?![0-9])"
        ),
    ),
    ("private SSH transfer dependency", SSH_TRANSFER_CODE),
)


def run_git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", os.fspath(ROOT), *args],
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def release_candidates() -> list[Path]:
    result = run_git(
        "ls-files", "--cached", "--others", "--exclude-standard", "-z"
    )
    paths = {
        Path(os.fsdecode(item))
        for item in result.stdout.split(b"\0")
        if item
    }
    return sorted(paths, key=lambda path: path.as_posix())


def scan_bytes(label: str, data: bytes) -> list[str]:
    findings: list[str] = []
    for fragment in PRIVATE_PATH_FRAGMENTS:
        if fragment in data:
            findings.append(f"{label}: private absolute filesystem path")
            break
    if re.search(rb"\b[A-Za-z]:\\(?:Users|data|datasets|workspace)\\", data):
        findings.append(f"{label}: private Windows filesystem path")
    if PRIVATE_KEY_MARKER in data:
        findings.append(f"{label}: private-key header")
    for rule_name, pattern in CONTENT_RULES:
        if pattern.search(data):
            findings.append(f"{label}: {rule_name}")
    return findings


def audit_files(paths: list[Path]) -> list[str]:
    findings: list[str] = []
    total_size = 0

    for relative_path, expected_hash in PINNED_ASSETS.items():
        absolute_path = ROOT / relative_path
        if not absolute_path.is_file():
            findings.append(f"{relative_path}: required reviewed asset is missing")
            continue
        actual_hash = hashlib.sha256(absolute_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            findings.append(f"{relative_path}: reviewed asset hash changed")

    for relative_path in paths:
        absolute_path = ROOT / relative_path
        label = relative_path.as_posix()

        if absolute_path.is_symlink():
            findings.append(f"{label}: symbolic links are not allowed in the release")
            continue
        if not absolute_path.is_file():
            findings.append(f"{label}: candidate is not a regular file")
            continue

        lower_parts = {part.lower() for part in relative_path.parts[:-1]}
        lower_name = relative_path.name.lower()
        if lower_name in FORBIDDEN_NAMES or lower_name.startswith(".env."):
            findings.append(f"{label}: forbidden credential/private filename")
        if lower_parts & FORBIDDEN_DIRECTORY_NAMES:
            findings.append(f"{label}: forbidden private artifact directory")
        if (
            relative_path.suffix.lower() in FORBIDDEN_SUFFIXES
            and relative_path not in PINNED_ASSETS
        ):
            findings.append(f"{label}: forbidden model/data/credential extension")

        size = absolute_path.stat().st_size
        total_size += size
        if size > MAX_FILE_BYTES:
            findings.append(f"{label}: exceeds the 5 MiB release-file limit")

        data = absolute_path.read_bytes()
        if b"\0" in data[:8192] and relative_path not in PINNED_ASSETS:
            findings.append(f"{label}: unreviewed binary file")
        elif relative_path not in PINNED_ASSETS:
            try:
                data.decode("utf-8")
            except UnicodeDecodeError:
                findings.append(f"{label}: release text is not valid UTF-8")
        if data.startswith(b"version https://git-lfs.github.com/spec/v1"):
            findings.append(f"{label}: Git LFS pointers are not allowed")
        findings.extend(scan_bytes(label, data))

    if total_size > MAX_RELEASE_BYTES:
        findings.append("release candidate exceeds the 10 MiB total-size limit")

    return findings


def audit_git_state() -> list[str]:
    findings: list[str] = []

    # Scan the exact staged blobs as well as the working-tree candidates. This
    # prevents a secret staged in the index from being hidden by a later safe
    # edit in the working tree before the commit hook runs.
    index_size = 0
    index_entries = run_git("ls-files", "--stage", "-z").stdout.split(b"\0")
    for entry in index_entries:
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        mode, object_id, stage = metadata.split()
        relative_path = Path(os.fsdecode(raw_path))
        label = f"{relative_path.as_posix()} (Git index)"

        if stage != b"0":
            findings.append(f"{label}: unresolved index stage")
            continue
        if mode == b"120000":
            findings.append(f"{label}: symbolic links are not allowed")
        if mode == b"160000":
            findings.append(f"{label}: Git submodules are not allowed")

        data = run_git("cat-file", "blob", os.fsdecode(object_id)).stdout
        index_size += len(data)
        if len(data) > MAX_FILE_BYTES:
            findings.append(f"{label}: exceeds the 5 MiB release-file limit")
        if relative_path in PINNED_ASSETS:
            actual_hash = hashlib.sha256(data).hexdigest()
            if actual_hash != PINNED_ASSETS[relative_path]:
                findings.append(f"{label}: reviewed asset hash changed")
        elif b"\0" in data[:8192]:
            findings.append(f"{label}: unreviewed binary file")
        else:
            try:
                data.decode("utf-8")
            except UnicodeDecodeError:
                findings.append(f"{label}: release text is not valid UTF-8")
        if data.startswith(b"version https://git-lfs.github.com/spec/v1"):
            findings.append(f"{label}: Git LFS pointers are not allowed")
        findings.extend(scan_bytes(label, data))

    if index_size > MAX_RELEASE_BYTES:
        findings.append("Git index exceeds the 10 MiB total-size limit")

    remotes = run_git("remote", "-v").stdout
    findings.extend(scan_bytes(".git remote configuration", remotes))

    revisions = run_git("rev-list", "--all", check=False).stdout.strip()
    if revisions:
        history = run_git(
            "log",
            "--all",
            "--format=fuller",
            "--patch",
            "--no-ext-diff",
            "--no-renames",
        ).stdout
        findings.extend(scan_bytes("reachable Git history", history))

    fsck = run_git("fsck", "--full", "--no-reflogs", "--unreachable", check=False)
    fsck_output = fsck.stdout + fsck.stderr
    if b"dangling " in fsck_output or b"unreachable " in fsck_output:
        findings.append(".git: dangling or unreachable objects require review and pruning")

    return findings


def main() -> int:
    paths = release_candidates()
    findings = audit_files(paths)
    findings.extend(audit_git_state())

    if findings:
        print("Release safety audit failed:", file=sys.stderr)
        for finding in sorted(set(findings)):
            print(f"- {finding}", file=sys.stderr)
        return 1

    print(
        f"Release safety audit passed: {len(paths)} candidate files; "
        f"{len(PINNED_ASSETS)} reviewed binary assets."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
