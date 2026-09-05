#!/usr/bin/env python3
"""Local, dependency-free human onset annotation tool.

Only raw decoded_u8.npy arrays are read. Source metadata stays in a private,
hash-bound manifest and is never returned by the browser API.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import hmac
import json
import mmap
import os
import secrets
import sqlite3
import struct
import threading
import time
import urllib.parse
import zlib
from contextlib import contextmanager
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

HERE = Path(__file__).resolve().parent
FPS = 16
WINDOW_START = 72
WINDOW_END = 161
FRAME_COUNT = WINDOW_END - WINDOW_START
COMMAND_FRAME = 96
VIEWS = ("left", "right")
VIEW_WIDTH = 1280
HEIGHT = 720
PRIVATE_KEYS = {
    "source_rollout_id", "source_manifest_sha256", "decoded_u8_sha256",
    "decoded_u8_path", "scene", "seed", "arm", "transition", "view", "cue",
    "window_start", "window_end", "command_frame", "duplicate_group",
    "action_protocol_sha256", "action_tensor_hashes", "known_answer",
}
RESPONSES = {"onset", "none", "uncertain", "technical_failure"}
COMMAND_LABELS = {
    "back": "Forward → Backward",
    "yaw_positive": "Forward → Camera yaw right",
    "canonical_forward": "Continue forward",
    "synthetic_onset": "Synthetic: motion begins",
    "synthetic_none": "Synthetic: no motion change",
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(chunk):
            digest.update(data)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def opaque_id(secret: bytes, *parts: object) -> str:
    message = "\0".join(map(str, parts)).encode()
    return hmac.new(secret, message, hashlib.sha256).hexdigest()[:24]


def load_bound_manifest(path: Path) -> dict[str, Any]:
    envelope = json.loads(path.read_text())
    if set(envelope) != {"schema_version", "manifest", "manifest_sha256"}:
        raise ValueError("private manifest envelope has unexpected keys")
    actual = hashlib.sha256(canonical_json(envelope["manifest"])).hexdigest()
    if not hmac.compare_digest(actual, envelope["manifest_sha256"]):
        raise ValueError("private manifest hash mismatch")
    validate_private_manifest(envelope["manifest"])
    return envelope


def validate_private_manifest(manifest: dict[str, Any]) -> None:
    required = {
        "purpose", "source_manifest_path", "source_manifest_sha256",
        "created_at", "protocol_sha256", "selection_rule", "items",
    }
    if not required.issubset(manifest):
        raise ValueError(f"manifest missing {sorted(required - set(manifest))}")
    if manifest["purpose"] != "tool_qc_validation":
        raise ValueError("current builder may only produce tool/QC validation manifests")
    ids: set[str] = set()
    for item in manifest["items"]:
        needed = {
            "opaque_id", "kind", "command_label", "source_type", "view",
            "window_start", "window_end", "command_frame", "duplicate_group",
        }
        if not needed.issubset(item):
            raise ValueError(f"item missing {sorted(needed - set(item))}")
        if item["opaque_id"] in ids:
            raise ValueError("duplicate opaque item id")
        ids.add(item["opaque_id"])
        if item["window_end"] - item["window_start"] != FRAME_COUNT:
            raise ValueError("unexpected frame count")
        if item["kind"] == "primary" and not item.get("primary_group"):
            raise ValueError("primary item lacks assignment group")
        if item["source_type"] == "raw_npy":
            if item["view"] not in VIEWS or not item.get("decoded_u8_sha256"):
                raise ValueError("raw item lacks source provenance")


def build_private_manifest(source_path: Path, output_path: Path, secret: bytes) -> dict[str, Any]:
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    root = source_path.parent
    protocol_hash = source.get("protocol_sha256")
    action_hash = source.get("action_protocol_sha256")
    if not protocol_hash or not action_hash:
        raise ValueError("supplied manifest lacks protocol/action hash binding")
    complete = [r for r in source.get("rollouts", []) if r.get("status") == "complete"]
    interventions = [
        r for r in complete
        if r.get("transition") in {"back", "yaw_positive"}
        and r.get("arm") == r.get("transition")
        and r.get("rollout_id", "").endswith("_d0")
        and r.get("raw_support_starts", {"0": 105, "1": 105}) in (
            {"0": 105, "1": 105}, {0: 105, 1: 105}
        )
    ]
    if not interventions:
        raise ValueError("no eligible canonical directional_d0 interventions")
    items: list[dict[str, Any]] = []

    def raw_item(r: dict[str, Any], view: str, kind: str, qc_type: str | None = None,
                 duplicate_group: str | None = None) -> dict[str, Any]:
        artifact = r["artifacts"]["decoded_u8"]
        path = (root / artifact["path"]).resolve()
        transition = r["transition"]
        group = f"{r['scene']}×{r['seed']}"
        ident = opaque_id(secret, r["rollout_id"], view, kind, qc_type or "", duplicate_group or "")
        return {
            "opaque_id": ident,
            "kind": kind,
            "qc_type": qc_type,
            "source_type": "raw_npy",
            "source_rollout_id": r["rollout_id"],
            "source_manifest_sha256": source_hash,
            "decoded_u8_path": str(path),
            "decoded_u8_sha256": artifact["sha256"],
            "scene": r["scene"],
            "seed": r["seed"],
            "arm": r["arm"],
            "transition": transition,
            "command_label": COMMAND_LABELS[transition],
            "view": view,
            "cue": transition,
            "window_start": WINDOW_START,
            "window_end": WINDOW_END,
            "command_frame": COMMAND_FRAME,
            "primary_group": group,
            "duplicate_group": duplicate_group or ident,
            "action_protocol_sha256": r["action_protocol_sha256"],
            "action_tensor_hashes": r.get("action_tensor_hashes", {}),
            "known_answer": None if kind == "primary" or qc_type == "delayed_repeat" else "none",
        }

    # Selection uses only manifest design/provenance fields, never detector scores.
    for rollout in sorted(interventions, key=lambda r: r["rollout_id"]):
        for view in VIEWS:
            items.append(raw_item(rollout, view, "primary"))

    controls = [
        r for r in complete
        if r.get("transition") == "canonical_forward"
        and r.get("arm") in {"control", "null_a", "null_b"}
    ]
    # Deterministic provenance-only sampling: one control and one null per scene, both views.
    qc_sources: list[dict[str, Any]] = []
    for scene in sorted({r["scene"] for r in controls}):
        scene_rows = sorted((r for r in controls if r["scene"] == scene), key=lambda r: r["rollout_id"])
        for arm_class in ("control", "null"):
            choices = [r for r in scene_rows if (r["arm"] == "control") == (arm_class == "control")]
            if choices:
                qc_sources.append(choices[0])
    for rollout in qc_sources:
        for view in VIEWS:
            item = raw_item(rollout, view, "qc", "canonical_null")
            # Present a plausible command; never disclose that this is QC.
            cue = ("back", "yaw_positive")[
                int(hashlib.sha256(f"{rollout['rollout_id']}:{view}".encode()).hexdigest(), 16) % 2
            ]
            item["cue"] = cue
            item["command_label"] = COMMAND_LABELS[cue]
            items.append(item)

    primaries = [i for i in items if i["kind"] == "primary"]
    # Four delayed exact repeats, stratified by transition/view and hidden from clients.
    for transition, view in (("back", "left"), ("back", "right"),
                             ("yaw_positive", "left"), ("yaw_positive", "right")):
        original = next(i for i in primaries if i["transition"] == transition and i["view"] == view)
        repeat = dict(original)
        repeat["opaque_id"] = opaque_id(secret, original["opaque_id"], "delayed_repeat")
        repeat["kind"] = "qc"
        repeat["qc_type"] = "delayed_repeat"
        repeat["duplicate_group"] = original["duplicate_group"]
        repeat["primary_group"] = original["primary_group"]
        items.append(repeat)

    for known, onset in (("onset", 32), ("none", None)):
        for variant in range(2):
            transition = f"synthetic_{known}"
            ident = opaque_id(secret, transition, variant)
            items.append({
                "opaque_id": ident, "kind": "qc", "qc_type": f"synthetic_{known}",
                "source_type": "synthetic", "source_rollout_id": None,
                "source_manifest_sha256": source_hash, "decoded_u8_path": None,
                "decoded_u8_sha256": None, "scene": "synthetic", "seed": variant,
                "arm": "synthetic", "transition": transition,
                "command_label": COMMAND_LABELS["back"], "view": "synthetic",
                "cue": transition, "window_start": WINDOW_START, "window_end": WINDOW_END,
                "command_frame": COMMAND_FRAME, "primary_group": f"synthetic×{known}×{variant}",
                "duplicate_group": ident, "action_protocol_sha256": action_hash,
                "action_tensor_hashes": {}, "known_answer": known,
                "known_onset_index": onset,
            })

    manifest = {
        "purpose": "tool_qc_validation",
        "study_status": "no_human_data_or_claim",
        "source_manifest_path": str(source_path.resolve()),
        "source_manifest_sha256": source_hash,
        "created_at": utc_now(),
        "protocol_sha256": sha256_file(HERE / "protocol.json"),
        "selection_rule": (
            "All complete back_d0/yaw_positive_d0 intervention views selected from "
            "provenance fields only; detector outcomes are neither loaded nor consulted."
        ),
        "items": items,
    }
    validate_private_manifest(manifest)
    envelope = {
        "schema_version": "rtwm-human-private-manifest-1",
        "manifest": manifest,
        "manifest_sha256": hashlib.sha256(canonical_json(manifest)).hexdigest(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(envelope) + b"\n")
    os.replace(temporary, output_path)
    return envelope


class NpyRGB:
    """Minimal read-only C-order uint8 NPY reader for [T,720,2560,3]."""
    def __init__(self, path: Path):
        self.path = path
        self.handle = path.open("rb")
        magic = self.handle.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError("not an NPY file")
        major, _minor = self.handle.read(2)
        header_len = struct.unpack("<H" if major == 1 else "<I",
                                   self.handle.read(2 if major == 1 else 4))[0]
        header = ast.literal_eval(self.handle.read(header_len).decode("latin1").strip())
        self.offset = self.handle.tell()
        self.shape = tuple(header["shape"])
        if header["descr"] not in ("|u1", "<u1") or header["fortran_order"]:
            raise ValueError("NPY must be C-order uint8")
        if len(self.shape) != 4 or self.shape[1:] != (HEIGHT, VIEW_WIDTH * 2, 3):
            raise ValueError(f"unexpected NPY shape {self.shape}")
        self.map = mmap.mmap(self.handle.fileno(), 0, access=mmap.ACCESS_READ)

    def view_frame(self, frame: int, view: str) -> bytes:
        if not 0 <= frame < self.shape[0] or view not in VIEWS:
            raise IndexError("frame/view out of range")
        full_row = VIEW_WIDTH * 2 * 3
        x_offset = 0 if view == "left" else VIEW_WIDTH * 3
        base = self.offset + frame * HEIGHT * full_row
        return b"".join(
            self.map[base + y * full_row + x_offset:base + y * full_row + x_offset + VIEW_WIDTH * 3]
            for y in range(HEIGHT)
        )

    def close(self) -> None:
        self.map.close()
        self.handle.close()


def png_rgb(width: int, height: int, rgb: bytes) -> bytes:
    if len(rgb) != width * height * 3:
        raise ValueError("RGB byte length mismatch")
    raw = b"".join(b"\0" + rgb[y * width * 3:(y + 1) * width * 3] for y in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 4)) + chunk(b"IEND", b""))


def synthetic_frame(index: int, onset: int | None) -> bytes:
    width, height = 640, 360
    shift = 0 if onset is None or index < onset else min(120, (index - onset + 1) * 4)
    row = bytearray()
    for x in range(width):
        band = ((x + shift) // 40) % 2
        row.extend((35 + band * 70, 65 + band * 45, 105 + band * 60))
    return png_rgb(width, height, bytes(row) * height)


class Store:
    def __init__(self, path: Path, items: list[dict[str, Any]], manifest_hash: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.items = {i["opaque_id"]: i for i in items}
        self.manifest_hash = manifest_hash
        self._init()

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=30000")
        return db

    def _init(self) -> None:
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
            CREATE TABLE IF NOT EXISTS annotators(
              id TEXT PRIMARY KEY, created_at TEXT NOT NULL, consented_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS assignments(
              id INTEGER PRIMARY KEY, annotator_id TEXT NOT NULL REFERENCES annotators(id),
              item_id TEXT NOT NULL, assigned_at REAL NOT NULL, lease_until REAL NOT NULL,
              completed_at REAL, UNIQUE(annotator_id,item_id));
            CREATE UNIQUE INDEX IF NOT EXISTS one_open_assignment
              ON assignments(annotator_id) WHERE completed_at IS NULL;
            CREATE TABLE IF NOT EXISTS playback(
              assignment_id INTEGER NOT NULL REFERENCES assignments(id),
              frame_index INTEGER NOT NULL, first_seen REAL NOT NULL,
              PRIMARY KEY(assignment_id,frame_index));
            CREATE TABLE IF NOT EXISTS responses(
              id INTEGER PRIMARY KEY, assignment_id INTEGER NOT NULL UNIQUE REFERENCES assignments(id),
              annotator_id TEXT NOT NULL, item_id TEXT NOT NULL, response TEXT NOT NULL,
              confidence INTEGER NOT NULL, onset_index INTEGER, source_frame INTEGER, lag_frames INTEGER,
              plays INTEGER NOT NULL, replays INTEGER NOT NULL, steps INTEGER NOT NULL,
              decision_ms INTEGER NOT NULL, hidden_ms INTEGER NOT NULL,
              visibility_changes INTEGER NOT NULL, preload_failures INTEGER NOT NULL,
              client_events_json TEXT NOT NULL, submitted_at TEXT NOT NULL,
              manifest_sha256 TEXT NOT NULL, UNIQUE(annotator_id,item_id));
            """)

    @contextmanager
    def immediate(self) -> Iterator[sqlite3.Connection]:
        db = self.connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.execute("COMMIT")
        except Exception:
            db.execute("ROLLBACK")
            raise
        finally:
            db.close()

    def register(self, annotator: str) -> None:
        if not (3 <= len(annotator) <= 80) or not all(c.isalnum() or c in "-_" for c in annotator):
            raise ValueError("invalid annotator code")
        now = utc_now()
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO annotators VALUES(?,?,?)", (annotator, now, now))

    def assign(self, annotator: str) -> sqlite3.Row | None:
        now = time.time()
        with self.immediate() as db:
            db.execute("DELETE FROM playback WHERE assignment_id IN "
                       "(SELECT id FROM assignments WHERE completed_at IS NULL AND lease_until<?)", (now,))
            db.execute("DELETE FROM assignments WHERE completed_at IS NULL AND lease_until<?", (now,))
            existing = db.execute(
                "SELECT * FROM assignments WHERE annotator_id=? AND completed_at IS NULL", (annotator,)
            ).fetchone()
            if existing:
                return existing
            done_ids = {r[0] for r in db.execute(
                "SELECT item_id FROM responses WHERE annotator_id=?", (annotator,))}
            done_primary_groups = {
                self.items[item_id].get("primary_group") for item_id in done_ids
                if item_id in self.items and self.items[item_id]["kind"] == "primary"
            }
            primary_done = sum(1 for item_id in done_ids if self.items[item_id]["kind"] == "primary")
            counts = dict(db.execute(
                "SELECT item_id,COUNT(*) FROM responses GROUP BY item_id").fetchall())
            candidates = []
            for item in self.items.values():
                if item["opaque_id"] in done_ids:
                    continue
                if item["kind"] == "primary" and item["primary_group"] in done_primary_groups:
                    continue
                if item.get("qc_type") == "delayed_repeat":
                    original_done = any(
                        self.items[x]["duplicate_group"] == item["duplicate_group"]
                        for x in done_ids if x in self.items
                    )
                    if primary_done < 4 or not original_done:
                        continue
                priority = 0 if (item.get("qc_type") or "").startswith("synthetic") and primary_done < 2 else 1
                if item.get("qc_type") == "delayed_repeat":
                    priority = 3
                candidates.append((priority, counts.get(item["opaque_id"], 0),
                                   hashlib.sha256(f"{annotator}:{item['opaque_id']}".encode()).hexdigest(), item))
            if not candidates:
                return None
            item = min(candidates, key=lambda x: x[:3])[3]
            cursor = db.execute(
                "INSERT INTO assignments(annotator_id,item_id,assigned_at,lease_until) VALUES(?,?,?,?)",
                (annotator, item["opaque_id"], now, now + 1800))
            return db.execute("SELECT * FROM assignments WHERE id=?", (cursor.lastrowid,)).fetchone()

    def log_frame(self, assignment_id: int, annotator: str, item_id: str, index: int) -> None:
        with self.connect() as db:
            row = db.execute(
                "SELECT 1 FROM assignments WHERE id=? AND annotator_id=? AND item_id=? AND completed_at IS NULL",
                (assignment_id, annotator, item_id)).fetchone()
            if not row:
                raise ValueError("invalid assignment")
            db.execute("INSERT OR IGNORE INTO playback VALUES(?,?,?)", (assignment_id, index, time.time()))

    def submit(self, assignment_id: int, annotator: str, payload: dict[str, Any]) -> dict[str, Any]:
        validate_response(payload)
        with self.immediate() as db:
            assignment = db.execute(
                "SELECT * FROM assignments WHERE id=? AND annotator_id=? AND completed_at IS NULL",
                (assignment_id, annotator)).fetchone()
            if not assignment:
                raise ValueError("assignment unavailable or already submitted")
            item = self.items[assignment["item_id"]]
            playback = db.execute(
                "SELECT COUNT(*) n,MIN(first_seen) lo,MAX(first_seen) hi FROM playback WHERE assignment_id=?",
                (assignment_id,)).fetchone()
            # All frames must have been requested over a near-real-time full pass.
            if playback["n"] != FRAME_COUNT or playback["hi"] - playback["lo"] < (FRAME_COUNT - 1) / FPS * 0.85:
                raise ValueError("complete 16-FPS playback is required before submission")
            onset = payload.get("onset_index") if payload["response"] == "onset" else None
            source_frame = item["window_start"] + onset if onset is not None else None
            lag = source_frame - item["command_frame"] if source_frame is not None else None
            values = (
                assignment_id, annotator, item["opaque_id"], payload["response"], payload["confidence"],
                onset, source_frame, lag, payload["plays"], payload["replays"], payload["steps"],
                payload["decision_ms"], payload["hidden_ms"], payload["visibility_changes"],
                payload["preload_failures"], json.dumps(payload.get("events", []), separators=(",", ":")),
                utc_now(), self.manifest_hash,
            )
            db.execute("""INSERT INTO responses(
              assignment_id,annotator_id,item_id,response,confidence,onset_index,source_frame,lag_frames,
              plays,replays,steps,decision_ms,hidden_ms,visibility_changes,preload_failures,
              client_events_json,submitted_at,manifest_sha256) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", values)
            db.execute("UPDATE assignments SET completed_at=? WHERE id=?", (time.time(), assignment_id))
            return {"ok": True, "recorded": True}


def validate_response(payload: dict[str, Any]) -> None:
    required = {
        "response", "confidence", "plays", "replays", "steps", "decision_ms",
        "hidden_ms", "visibility_changes", "preload_failures", "events",
    }
    if not required.issubset(payload):
        raise ValueError(f"missing response fields: {sorted(required - set(payload))}")
    allowed = required | {"onset_index"}
    if set(payload) - allowed:
        raise ValueError(f"unexpected response fields: {sorted(set(payload) - allowed)}")
    if payload["response"] not in RESPONSES:
        raise ValueError("invalid response")
    if type(payload["confidence"]) is not int or not 1 <= payload["confidence"] <= 5:
        raise ValueError("confidence must be 1..5")
    onset = payload.get("onset_index")
    if payload["response"] == "onset" and (not isinstance(onset, int) or not 0 <= onset < FRAME_COUNT):
        raise ValueError("onset response requires a valid onset_index")
    if payload["response"] != "onset" and onset is not None:
        raise ValueError("non-onset response cannot include onset_index")
    for key in ("plays", "replays", "steps", "decision_ms", "hidden_ms",
                "visibility_changes", "preload_failures"):
        if type(payload[key]) is not int or payload[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if payload["plays"] < 1 or not isinstance(payload["events"], list):
        raise ValueError("play metadata invalid")


class FrameCache:
    def __init__(self, limit: int = 4):
        self.limit = limit
        self.readers: dict[str, NpyRGB] = {}
        self.lock = threading.Lock()

    def get(self, item: dict[str, Any], index: int) -> bytes:
        if item["source_type"] == "synthetic":
            return synthetic_frame(index, item.get("known_onset_index"))
        path = item["decoded_u8_path"]
        with self.lock:
            if path not in self.readers:
                if len(self.readers) >= self.limit:
                    _, old = self.readers.popitem()
                    old.close()
                self.readers[path] = NpyRGB(Path(path))
            rgb = self.readers[path].view_frame(item["window_start"] + index, item["view"])
        return png_rgb(VIEW_WIDTH, HEIGHT, rgb)


def public_item(item: dict[str, Any], assignment_id: int) -> dict[str, Any]:
    return {
        "item_id": item["opaque_id"], "assignment_id": assignment_id,
        "task": "first_visible_commanded_response", "command_label": item["command_label"],
        "frame_count": FRAME_COUNT, "fps": FPS,
    }


def make_handler(store: Store, assets: Path, cache: FrameCache) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "RTWMHumanTool/1"

        def _json(self, status: int, value: Any) -> None:
            body = canonical_json(value)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 1_000_000:
                raise ValueError("request too large")
            return json.loads(self.rfile.read(length) or b"{}")

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/":
                return self._asset("index.html")
            if parsed.path.startswith("/static/"):
                return self._asset(parsed.path.removeprefix("/static/"))
            if parsed.path == "/api/next":
                try:
                    annotator = urllib.parse.parse_qs(parsed.query).get("annotator", [""])[0]
                    store.register(annotator)
                    assignment = store.assign(annotator)
                    if assignment is None:
                        return self._json(200, {"done": True})
                    return self._json(200, public_item(store.items[assignment["item_id"]], assignment["id"]))
                except ValueError as exc:
                    return self._json(400, {"error": str(exc)})
            if parsed.path.startswith("/api/frame/"):
                try:
                    _, _, _, item_id, index_text = parsed.path.split("/")
                    query = urllib.parse.parse_qs(parsed.query)
                    annotator = query.get("annotator", [""])[0]
                    assignment_id = int(query.get("assignment", ["0"])[0])
                    index = int(index_text)
                    if item_id not in store.items or not 0 <= index < FRAME_COUNT:
                        raise ValueError("unknown frame")
                    store.log_frame(assignment_id, annotator, item_id, index)
                    body = cache.get(store.items[item_id], index)
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "private, max-age=300")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.end_headers()
                    return self.wfile.write(body)
                except (ValueError, IndexError):
                    return self._json(400, {"error": "unknown or unauthorized frame"})
                except OSError:
                    return self._json(500, {"error": "frame media unavailable"})
            self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/api/submit":
                return self._json(404, {"error": "not found"})
            try:
                body = self._body()
                result = store.submit(int(body.pop("assignment_id")), body.pop("annotator"), body)
                return self._json(200, result)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                return self._json(400, {"error": str(exc)})

        def _asset(self, name: str) -> None:
            if name not in {"index.html", "app.js", "style.css"}:
                return self._json(404, {"error": "not found"})
            body = (assets / name).read_bytes()
            mime = {"html": "text/html; charset=utf-8", "js": "text/javascript; charset=utf-8",
                    "css": "text/css; charset=utf-8"}[name.rsplit(".", 1)[1]]
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; img-src 'self'; script-src 'self'; style-src 'self'")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: object) -> None:
            print(f"{self.address_string()} [{self.log_date_time_string()}] {fmt % args}")
    return Handler


def export_responses(store: Store, private: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError("export destination must not exist (exports are immutable)")
    output_dir.mkdir(parents=True)
    item_by_id = {i["opaque_id"]: i for i in private["manifest"]["items"]}
    with store.connect() as db:
        rows = [dict(r) for r in db.execute("SELECT * FROM responses ORDER BY id")]
    enriched = []
    for row in rows:
        item = item_by_id[row["item_id"]]
        enriched.append({**row, **{k: item.get(k) for k in (
            "source_rollout_id", "decoded_u8_sha256", "scene", "seed", "arm",
            "transition", "view", "cue", "window_start", "window_end",
            "command_frame", "duplicate_group", "qc_type", "known_answer")}})
    jsonl = output_dir / "responses.jsonl"
    jsonl.write_bytes(b"".join(canonical_json(r) + b"\n" for r in enriched))
    csv_path = output_dir / "responses.csv"
    fields = sorted({key for row in enriched for key in row})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for row in enriched:
            writer.writerow(row)
    manifest = {
        "schema_version": "rtwm-human-export-1", "created_at": utc_now(),
        "row_count": len(enriched), "private_manifest_sha256": private["manifest_sha256"],
        "files": {p.name: sha256_file(p) for p in (jsonl, csv_path)},
    }
    manifest["export_sha256"] = hashlib.sha256(canonical_json(manifest)).hexdigest()
    (output_dir / "export_manifest.json").write_bytes(canonical_json(manifest) + b"\n")
    for path in output_dir.iterdir():
        path.chmod(0o444)
    output_dir.chmod(0o555)
    return manifest


def cmd_build(args: argparse.Namespace) -> None:
    secret_path = Path(args.secret)
    secret_path.parent.mkdir(parents=True, exist_ok=True)
    if secret_path.exists():
        secret = secret_path.read_bytes()
    else:
        secret = secrets.token_bytes(32)
        secret_path.write_bytes(secret)
        secret_path.chmod(0o600)
    result = build_private_manifest(Path(args.source), Path(args.output), secret)
    counts: dict[str, int] = {}
    for item in result["manifest"]["items"]:
        key = item.get("qc_type") or item["kind"]
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({"items": len(result["manifest"]["items"]), "counts": counts,
                      "manifest_sha256": result["manifest_sha256"]}, indent=2))


def cmd_serve(args: argparse.Namespace) -> None:
    private = load_bound_manifest(Path(args.manifest))
    store = Store(Path(args.db), private["manifest"]["items"], private["manifest_sha256"])
    server = ThreadingHTTPServer(("127.0.0.1", args.port),
                                 make_handler(store, HERE / "static", FrameCache()))
    print(f"Serving http://127.0.0.1:{server.server_port} (localhost only)")
    server.serve_forever()


def cmd_validate(args: argparse.Namespace) -> None:
    private = load_bound_manifest(Path(args.manifest))
    checked = 0
    verified_hashes: dict[str, str] = {}
    for item in private["manifest"]["items"]:
        if item["source_type"] != "raw_npy":
            continue
        path = Path(item["decoded_u8_path"])
        if args.hash_sources:
            if str(path) not in verified_hashes:
                verified_hashes[str(path)] = sha256_file(path)
            actual = verified_hashes[str(path)]
            if actual != item["decoded_u8_sha256"]:
                raise ValueError(f"source hash mismatch: {path}")
        reader = NpyRGB(path)
        reader.view_frame(WINDOW_START, item["view"])
        reader.view_frame(WINDOW_END - 1, item["view"])
        reader.close()
        checked += 1
    print(json.dumps({"valid": True, "items": len(private["manifest"]["items"]),
                      "raw_item_checks": checked,
                      "unique_source_hashes_checked": len(verified_hashes)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(required=True)
    build = commands.add_parser("build")
    build.add_argument("--source", required=True)
    build.add_argument("--output", default=str(HERE / "runtime/private_manifest.json"))
    build.add_argument("--secret", default=str(HERE / "runtime/secret.key"))
    build.set_defaults(func=cmd_build)
    serve = commands.add_parser("serve")
    serve.add_argument("--manifest", default=str(HERE / "runtime/private_manifest.json"))
    serve.add_argument("--db", default=str(HERE / "runtime/annotations.sqlite3"))
    serve.add_argument("--port", type=int, default=8765)
    serve.set_defaults(func=cmd_serve)
    validate = commands.add_parser("validate")
    validate.add_argument("--manifest", default=str(HERE / "runtime/private_manifest.json"))
    validate.add_argument("--hash-sources", action="store_true")
    validate.set_defaults(func=cmd_validate)
    export = commands.add_parser("export")
    export.add_argument("--manifest", default=str(HERE / "runtime/private_manifest.json"))
    export.add_argument("--db", default=str(HERE / "runtime/annotations.sqlite3"))
    export.add_argument("--output", required=True)
    export.set_defaults(func=lambda a: print(json.dumps(export_responses(
        Store(Path(a.db), (p := load_bound_manifest(Path(a.manifest)))["manifest"]["items"],
              p["manifest_sha256"]), p, Path(a.output)), indent=2)))
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
