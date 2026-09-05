import hashlib
import http.client
import json
import os
import sqlite3
import struct
import tempfile
import threading
import time
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import human_tool as ht


def sparse_npy(path: Path, frames: int = 162) -> tuple[int, int]:
    header_dict = {"descr": "|u1", "fortran_order": False,
                   "shape": (frames, ht.HEIGHT, ht.VIEW_WIDTH * 2, 3)}
    text = repr(header_dict)
    padding = 16 - ((10 + len(text) + 1) % 16)
    header = (text + " " * padding + "\n").encode("latin1")
    with path.open("wb") as handle:
        handle.write(b"\x93NUMPY" + bytes((1, 0)) + struct.pack("<H", len(header)) + header)
        offset = handle.tell()
        size = frames * ht.HEIGHT * ht.VIEW_WIDTH * 2 * 3
        handle.seek(offset + size - 1)
        handle.write(b"\0")
        # Distinguish left/right and adjacent source frames at one pixel.
        row = ht.VIEW_WIDTH * 2 * 3
        for frame, left, right in ((72, b"\x01\x02\x03", b"\x04\x05\x06"),
                                   (73, b"\x07\x08\x09", b"\x0a\x0b\x0c")):
            base = offset + frame * ht.HEIGHT * row
            handle.seek(base)
            handle.write(left)
            handle.seek(base + ht.VIEW_WIDTH * 3)
            handle.write(right)
    return offset, size


def rollout(root: Path, rollout_id: str, scene: str, seed: int, arm: str,
            transition: str, npy: Path) -> dict:
    return {
        "rollout_id": rollout_id, "scene": scene, "seed": seed, "arm": arm,
        "transition": transition, "status": "complete",
        "raw_support_starts": {"0": 105, "1": 105} if transition != "canonical_forward" else None,
        "action_protocol_sha256": "a" * 64,
        "action_tensor_hashes": {"post_keyboard_sha256": "b" * 64},
        "artifacts": {"decoded_u8": {
            "path": os.path.relpath(npy, root), "sha256": "c" * 64,
        }},
    }


def make_source(directory: Path) -> tuple[Path, Path]:
    npy = directory / "raw.npy"
    sparse_npy(npy)
    source = {
        "protocol_sha256": "d" * 64, "action_protocol_sha256": "a" * 64,
        "rollouts": [
            rollout(directory, "s1_back_d0", "sceneA", 1, "back", "back", npy),
            rollout(directory, "s1_yaw_d0", "sceneA", 1, "yaw_positive", "yaw_positive", npy),
            rollout(directory, "s2_control", "sceneA", 2, "control", "canonical_forward", npy),
            rollout(directory, "s3_null", "sceneA", 3, "null_a", "canonical_forward", npy),
        ],
    }
    path = directory / "source.json"
    path.write_text(json.dumps(source))
    return path, npy


def direct_complete(store: ht.Store, assignment: sqlite3.Row) -> None:
    with store.connect() as db:
        db.execute("""INSERT INTO responses(
          assignment_id,annotator_id,item_id,response,confidence,onset_index,source_frame,lag_frames,
          plays,replays,steps,decision_ms,hidden_ms,visibility_changes,preload_failures,
          client_events_json,submitted_at,manifest_sha256)
          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
          (assignment["id"], assignment["annotator_id"], assignment["item_id"], "none", 3,
           None, None, None, 1, 0, 0, 6000, 0, 0, 0, "[]", ht.utc_now(), store.manifest_hash))
        db.execute("UPDATE assignments SET completed_at=? WHERE id=?", (time.time(), assignment["id"]))


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source, self.npy = make_source(self.root)
        self.private_path = self.root / "private.json"
        self.envelope = ht.build_private_manifest(self.source, self.private_path, b"k" * 32)
        self.items = self.envelope["manifest"]["items"]

    def tearDown(self):
        self.temp.cleanup()

    def test_manifest_selection_counts_and_independence(self):
        primary = [i for i in self.items if i["kind"] == "primary"]
        self.assertEqual(4, len(primary))
        self.assertEqual({"back", "yaw_positive"}, {i["transition"] for i in primary})
        self.assertEqual({"left", "right"}, {i["view"] for i in primary})
        self.assertEqual(4, sum((i.get("qc_type") or "").startswith("synthetic") for i in self.items))
        self.assertEqual(4, sum(i.get("qc_type") == "delayed_repeat" for i in self.items))
        source_text = self.source.read_text()
        self.assertNotIn("score", source_text)
        self.assertIn("detector outcomes are neither loaded", self.envelope["manifest"]["selection_rule"])

    def test_private_manifest_hash_binding(self):
        loaded = ht.load_bound_manifest(self.private_path)
        self.assertEqual(self.envelope["manifest_sha256"], loaded["manifest_sha256"])
        value = json.loads(self.private_path.read_text())
        value["manifest"]["items"][0]["seed"] = 999
        self.private_path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            ht.load_bound_manifest(self.private_path)

    def test_public_item_has_no_private_metadata(self):
        item = next(i for i in self.items if i["kind"] == "primary")
        payload = ht.public_item(item, 7)
        encoded = json.dumps(payload)
        self.assertTrue(set(payload) <= {"item_id", "assignment_id", "task", "command_label",
                                        "frame_count", "fps"})
        for key in ht.PRIVATE_KEYS:
            self.assertNotIn(key, encoded)
        self.assertNotIn(item["scene"], encoded)
        self.assertNotIn(str(self.npy), encoded)

    def test_response_schema_validation(self):
        valid = {"response": "onset", "onset_index": 24, "confidence": 4, "plays": 1,
                 "replays": 0, "steps": 2, "decision_ms": 7000, "hidden_ms": 0,
                 "visibility_changes": 0, "preload_failures": 0, "events": []}
        ht.validate_response(valid)
        for bad in ({**valid, "confidence": 0}, {**valid, "onset_index": 89},
                    {**valid, "response": "none"}, {**valid, "plays": 0},
                    {**valid, "private_guess": "sceneA"}):
            with self.assertRaises(ValueError):
                ht.validate_response(bad)
        schema = json.loads((ht.HERE / "schema.json").read_text())
        self.assertEqual(set(ht.RESPONSES), set(schema["properties"]["response"]["enum"]))

    def test_npy_frame_index_and_view(self):
        reader = ht.NpyRGB(self.npy)
        try:
            self.assertEqual(b"\x01\x02\x03", reader.view_frame(72, "left")[:3])
            self.assertEqual(b"\x04\x05\x06", reader.view_frame(72, "right")[:3])
            self.assertEqual(b"\x07\x08\x09", reader.view_frame(73, "left")[:3])
            with self.assertRaises(IndexError):
                reader.view_frame(999, "left")
        finally:
            reader.close()

    def test_frame_hash_and_png_signature(self):
        reader = ht.NpyRGB(self.npy)
        try:
            frame = reader.view_frame(72, "left")
            right = reader.view_frame(72, "right")
            adjacent = reader.view_frame(73, "left")
            self.assertEqual(ht.VIEW_WIDTH * ht.HEIGHT * 3, len(frame))
            self.assertNotEqual(hashlib.sha256(frame).digest(), hashlib.sha256(right).digest())
            self.assertNotEqual(hashlib.sha256(frame).digest(), hashlib.sha256(adjacent).digest())
            png = ht.png_rgb(ht.VIEW_WIDTH, ht.HEIGHT, frame)
            self.assertEqual(b"\x89PNG\r\n\x1a\n", png[:8])
            self.assertIn(b"IHDR", png[:40])
        finally:
            reader.close()

    def test_sqlite_uniqueness_and_lease_recovery(self):
        store = ht.Store(self.root / "db.sqlite3", self.items, self.envelope["manifest_sha256"])
        store.register("ann_1")
        first = store.assign("ann_1")
        self.assertEqual(first["id"], store.assign("ann_1")["id"])
        with store.connect() as db:
            db.execute("UPDATE assignments SET lease_until=0 WHERE id=?", (first["id"],))
        recovered = store.assign("ann_1")
        self.assertGreater(recovered["assigned_at"], first["assigned_at"])
        with store.connect() as db:
            self.assertEqual(1, db.execute(
                "SELECT COUNT(*) FROM assignments WHERE annotator_id=? AND completed_at IS NULL",
                ("ann_1",)).fetchone()[0])
        with self.assertRaises(sqlite3.IntegrityError), store.connect() as db:
            db.execute("INSERT INTO assignments(annotator_id,item_id,assigned_at,lease_until) VALUES(?,?,?,?)",
                       ("ann_1", recovered["item_id"], time.time(), time.time() + 10))

    def test_assignment_balance_and_group_constraints(self):
        store = ht.Store(self.root / "assign.sqlite3", self.items, self.envelope["manifest_sha256"])
        for annotator in ("alice", "bob", "cara", "dave"):
            store.register(annotator)
            for _ in range(8):
                assignment = store.assign(annotator)
                if not assignment:
                    break
                direct_complete(store, assignment)
        with store.connect() as db:
            completed = dict(db.execute(
                "SELECT item_id,COUNT(*) FROM responses GROUP BY item_id").fetchall())
        counts = [completed.get(i["opaque_id"], 0) for i in self.items if i["kind"] == "primary"]
        self.assertLessEqual(max(counts) - min(counts), 1)
        for annotator in ("alice", "bob", "cara", "dave"):
            with store.connect() as db:
                ids = [r[0] for r in db.execute(
                    "SELECT item_id FROM responses WHERE annotator_id=?", (annotator,))]
            groups = [store.items[x]["primary_group"] for x in ids if store.items[x]["kind"] == "primary"]
            self.assertEqual(len(groups), len(set(groups)))

    def test_delayed_repeat_not_early(self):
        store = ht.Store(self.root / "delay.sqlite3", self.items, self.envelope["manifest_sha256"])
        store.register("repeat_tester")
        seen = []
        for _ in range(4):
            assignment = store.assign("repeat_tester")
            seen.append(store.items[assignment["item_id"]].get("qc_type"))
            direct_complete(store, assignment)
        self.assertNotIn("delayed_repeat", seen)

    def test_submission_derives_source_frame_and_lag(self):
        primary = next(i for i in self.items if i["kind"] == "primary")
        only = [primary]
        store = ht.Store(self.root / "submit.sqlite3", only, self.envelope["manifest_sha256"])
        store.register("annotator")
        assignment = store.assign("annotator")
        with store.connect() as db:
            base = time.time() - 6
            db.executemany("INSERT INTO playback VALUES(?,?,?)",
                           [(assignment["id"], i, base + i / ht.FPS) for i in range(ht.FRAME_COUNT)])
        payload = {"response": "onset", "onset_index": 30, "confidence": 5, "plays": 1,
                   "replays": 0, "steps": 3, "decision_ms": 8000, "hidden_ms": 10,
                   "visibility_changes": 2, "preload_failures": 0, "events": []}
        store.submit(assignment["id"], "annotator", payload)
        with store.connect() as db:
            row = db.execute("SELECT * FROM responses").fetchone()
        self.assertEqual(102, row["source_frame"])
        self.assertEqual(6, row["lag_frames"])
        with self.assertRaisesRegex(ValueError, "already submitted"):
            store.submit(assignment["id"], "annotator", payload)

    def test_submission_rejects_incomplete_playback(self):
        store = ht.Store(self.root / "reject.sqlite3", self.items[:1], self.envelope["manifest_sha256"])
        store.register("annotator")
        assignment = store.assign("annotator")
        payload = {"response": "none", "onset_index": None, "confidence": 3, "plays": 1,
                   "replays": 0, "steps": 0, "decision_ms": 1000, "hidden_ms": 0,
                   "visibility_changes": 0, "preload_failures": 0, "events": []}
        with self.assertRaisesRegex(ValueError, "playback"):
            store.submit(assignment["id"], "annotator", payload)

    def test_immutable_hash_bound_export(self):
        item = self.items[0]
        store = ht.Store(self.root / "export.sqlite3", [item], self.envelope["manifest_sha256"])
        store.register("exporter")
        assignment = store.assign("exporter")
        direct_complete(store, assignment)
        out = self.root / "export"
        manifest = ht.export_responses(store, self.envelope, out)
        self.assertEqual(1, manifest["row_count"])
        self.assertEqual(manifest["files"]["responses.jsonl"], ht.sha256_file(out / "responses.jsonl"))
        with self.assertRaises(FileExistsError):
            ht.export_responses(store, self.envelope, out)

    def test_server_api_and_leak_surface(self):
        synthetic = next(i for i in self.items if i.get("qc_type") == "synthetic_onset")
        store = ht.Store(self.root / "server.sqlite3", [synthetic], self.envelope["manifest_sha256"])
        server = ht.ThreadingHTTPServer(("127.0.0.1", 0),
                                        ht.make_handler(store, ht.HERE / "static", ht.FrameCache()))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
            connection.request("GET", "/api/next?annotator=api_user")
            response = connection.getresponse()
            value = json.loads(response.read())
            self.assertEqual(200, response.status)
            self.assertNotIn("scene", value)
            self.assertNotIn("qc", json.dumps(value))
            self.assertNotIn("synthetic", json.dumps(value).lower())
            url = f"/api/frame/{value['item_id']}/0?annotator=api_user&assignment={value['assignment_id']}"
            connection.request("GET", url)
            frame = connection.getresponse()
            self.assertEqual("image/png", frame.getheader("Content-Type"))
            self.assertEqual(b"\x89PNG\r\n\x1a\n", frame.read()[:8])
            body = json.dumps({"annotator": "api_user", "assignment_id": value["assignment_id"],
                               "response": "none", "onset_index": None, "confidence": 3,
                               "plays": 1, "replays": 0, "steps": 0, "decision_ms": 100,
                               "hidden_ms": 0, "visibility_changes": 0,
                               "preload_failures": 0, "events": []})
            connection.request("POST", "/api/submit", body, {"Content-Type": "application/json"})
            rejected = connection.getresponse()
            error = json.loads(rejected.read())
            self.assertEqual(400, rejected.status)
            self.assertIn("playback", error["error"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
