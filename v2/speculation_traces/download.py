"""Resumable, action-only downloader for the frozen VPT protocol."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from common import (
    DEFAULT_PROTOCOL,
    atomic_write_json,
    load_json,
    protocol_sha256,
    selected_episodes,
    sha256_file,
)

USER_AGENT = "rtwm-vpt-traces/1.0 (action-only research download)"


def _assert_action_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"refusing non-HTTPS URL: {url}")
    if parsed.netloc != "openaipublic.blob.core.windows.net":
        raise ValueError(f"refusing unexpected host: {parsed.netloc}")
    if not parsed.path.endswith(".jsonl") or ".mp4" in parsed.path:
        raise ValueError(f"refusing non-action URL: {url}")


def _count_valid_jsonl(path: Path) -> int:
    for encoding in ("utf-8", "windows-1252"):
        count = 0
        try:
            with path.open("r", encoding=encoding) as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"{path}:{line_number}: invalid JSON"
                        ) from error
                    if not isinstance(value, dict):
                        raise ValueError(
                            f"{path}:{line_number}: expected JSON object"
                        )
                    count += 1
            return count
        except UnicodeDecodeError:
            if encoding == "windows-1252":
                raise
    raise AssertionError("unreachable encoding fallback")


def _download_once(
    url: str,
    part_path: Path,
    protocol_hash: str,
    timeout: float,
) -> None:
    meta_path = part_path.with_suffix(part_path.suffix + ".meta.json")
    expected_meta = {"protocol_sha256": protocol_hash, "url": url}
    if part_path.exists():
        if not meta_path.exists() or load_json(meta_path) != expected_meta:
            part_path.unlink()
            meta_path.unlink(missing_ok=True)
    if not part_path.exists():
        atomic_write_json(meta_path, expected_meta)

    offset = part_path.stat().st_size if part_path.exists() else 0
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = response.getcode()
        content_range = response.headers.get("Content-Range", "")
        resume_ok = status == 206 and content_range.startswith(f"bytes {offset}-")
        mode = "ab" if offset and resume_ok else "wb"
        with part_path.open(mode) as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())


def download_episode(
    episode: dict[str, Any],
    output_dir: Path,
    protocol_hash: str,
    retries: int,
    timeout: float,
) -> dict[str, Any]:
    url = episode["source_url"]
    _assert_action_url(url)
    target = (
        output_dir
        / episode["split"]
        / f"{int(episode['episode_id']):06d}.jsonl"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    part_path = target.with_suffix(target.suffix + ".part")
    meta_path = part_path.with_suffix(part_path.suffix + ".meta.json")

    if target.exists():
        count = _count_valid_jsonl(target)
        if count == episode["frame_count"]:
            return _entry(episode, target, count)
        target.unlink()

    if part_path.exists():
        try:
            count = _count_valid_jsonl(part_path)
        except (UnicodeDecodeError, ValueError):
            count = -1
        if count == episode["frame_count"] and part_path.stat().st_size > 0:
            os.replace(part_path, target)
            meta_path.unlink(missing_ok=True)
            return _entry(episode, target, count)

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            _download_once(url, part_path, protocol_hash, timeout)
            count = _count_valid_jsonl(part_path)
            if count != episode["frame_count"]:
                raise ValueError(
                    f"expected {episode['frame_count']} JSONL records, got {count}"
                )
            if part_path.stat().st_size == 0:
                raise ValueError("downloaded file is empty")
            os.replace(part_path, target)
            meta_path.unlink(missing_ok=True)
            return _entry(episode, target, count)
        except (
            OSError,
            ValueError,
            urllib.error.HTTPError,
            urllib.error.URLError,
        ) as error:
            last_error = error
            if isinstance(error, urllib.error.HTTPError) and error.code in (404, 403):
                break
            if attempt < retries:
                delay = min(30.0, 2.0**attempt) + random.Random(
                    f"{episode['split']}:{episode['episode_id']}:{attempt}"
                ).random()
                time.sleep(delay)
    raise RuntimeError(
        f"failed {episode['split']}:{episode['episode_id']} ({url}): {last_error}"
    )


def _entry(
    episode: dict[str, Any], target: Path, jsonl_records: int
) -> dict[str, Any]:
    return {
        "split": episode["split"],
        "episode_id": episode["episode_id"],
        "source_url": episode["source_url"],
        "source_relpath": episode["source_relpath"],
        "local_path": str(target.resolve()),
        "bytes": target.stat().st_size,
        "jsonl_records": jsonl_records,
        "sha256": sha256_file(target),
    }


def download_protocol(
    protocol_path: Path,
    output_dir: Path,
    manifest_path: Path,
    splits: tuple[str, ...],
    limit: int | None,
    workers: int,
    retries: int,
    timeout: float,
) -> tuple[dict[str, Any], list[str]]:
    protocol = load_json(protocol_path)
    protocol_hash = protocol_sha256(protocol_path)
    episodes = selected_episodes(protocol, splits)
    if limit is not None:
        episodes = episodes[:limit]

    existing: dict[tuple[str, int], dict[str, Any]] = {}
    if manifest_path.exists():
        old = load_json(manifest_path)
        if old["protocol_sha256"] != protocol_hash:
            raise ValueError("existing download manifest belongs to another protocol")
        existing = {
            (entry["split"], int(entry["episode_id"])): entry
            for entry in old["files"]
        }

    results: list[dict[str, Any]] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                download_episode,
                episode,
                output_dir,
                protocol_hash,
                retries,
                timeout,
            ): episode
            for episode in episodes
        }
        for future in as_completed(futures):
            episode = futures[future]
            try:
                results.append(future.result())
            except Exception as error:  # retain all failures in one run report
                errors.append(
                    f"{episode['split']}:{episode['episode_id']}: {error}"
                )

    for entry in results:
        existing[(entry["split"], int(entry["episode_id"]))] = entry
    files = sorted(existing.values(), key=lambda item: (item["split"], item["episode_id"]))
    manifest = {
        "schema_version": 1,
        "protocol_sha256": protocol_hash,
        "files": files,
        "summary": {
            "downloaded_file_count": len(files),
            "downloaded_bytes": sum(int(entry["bytes"]) for entry in files),
        },
    }
    atomic_write_json(manifest_path, manifest)
    return manifest, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=Path("raw"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("manifests/downloads.json")
    )
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    splits = ("train", "test") if args.split == "all" else (args.split,)
    manifest, errors = download_protocol(
        args.protocol,
        args.output_dir,
        args.manifest,
        splits,
        args.limit,
        args.workers,
        args.retries,
        args.timeout,
    )
    print(json.dumps({**manifest["summary"], "errors": errors}, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

