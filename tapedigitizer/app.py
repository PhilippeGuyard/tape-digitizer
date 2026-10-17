"""The Tape Digitizer window: a small local web app.

    python -m tapedigitizer app

Starts a server on this Mac only (127.0.0.1) and opens the page in a browser
window. The page polls /api/state for meters and progress; everything else is
plain JSON requests. Audio work is the same code the command line uses.
"""

from __future__ import annotations

import csv
import json
import mimetypes
import queue
import re
import shutil
import subprocess
import threading
import time
import traceback
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import cli, splitter
from .recorder import RecorderSettings, State, find_device, list_inputs, parse_inputs, record_side

STATIC = Path(__file__).parent / "static"
SIDE_RE = re.compile(r"^(.+) - Side ([A-Za-z0-9]+)$")


def db(x: float) -> float:
    return float(20 * np.log10(max(x, 1e-6)))


class Engine:
    """Owns the audio input and the record, split and save job for one side at a time."""

    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.lock = threading.Lock()
        self.stream = None
        self.device = None
        self.inputs: tuple[int, ...] = (1, 2)
        self.source_error = ""
        self.meter: list[dict] = []
        self.rec_q: queue.Queue | None = None
        self.status = "idle"  # idle, waiting, recording, splitting, saving, done, error
        self.message = ""
        self.side = ""
        self.job = ""  # "record" or "edit"
        self.elapsed = 0.0
        self.silent_for = 0.0
        self.live_wave: list[float] = []
        self._wave_acc: list[float] = []
        self.stop_requested = False
        self.settings = RecorderSettings()
        self.load_source()

    # ---- audio source -------------------------------------------------

    def load_source(self):
        cfg = cli.load_config()
        dev = None
        if cfg.get("device_name"):
            dev = next((i for i, n, _, _ in list_inputs() if n == cfg["device_name"]), None)
        if dev is None:
            dev = find_device()
        try:
            inputs = parse_inputs(cfg.get("inputs", "1,2"))
        except ValueError:
            inputs = (1, 2)
        self.open_source(dev, inputs, save=False)

    def open_source(self, device, inputs, save=True):
        import sounddevice as sd

        if self.busy:
            raise RuntimeError("Can't change the input while a side is being recorded.")
        if self.stream is not None:
            self.stream.close()
            self.stream = None
        self.device, self.inputs, self.source_error = device, tuple(inputs), ""
        cols = [i - 1 for i in self.inputs]

        def callback(indata, frames, t, status):
            block = indata[:, cols]
            self.meter = [
                {"input": inp, "db": db(float(np.sqrt(np.mean(block[:, k] ** 2)))), "peak": db(float(np.max(np.abs(block[:, k]))))}
                for k, inp in enumerate(self.inputs)
            ]
            q = self.rec_q
            if q is not None:
                q.put(block.copy())

        try:
            if device is not None and max(self.inputs) > sd.query_devices(device)["max_input_channels"]:
                raise ValueError(f"{sd.query_devices(device)['name']} doesn't have input {max(self.inputs)}")
            self.stream = sd.InputStream(
                samplerate=self.settings.samplerate,
                channels=max(self.inputs),
                device=device,
                dtype="float32",
                blocksize=int(self.settings.samplerate * self.settings.block_seconds),
                callback=callback,
            )
            self.stream.start()
        except Exception as e:  # device unplugged, wrong rate, etc.
            self.stream = None
            self.source_error = str(e)
            self.meter = []
        if save:
            cfg = cli.load_config()
            names = {i: n for i, n, _, _ in list_inputs()}
            cfg.update(device=device, device_name=names.get(device), inputs=",".join(map(str, self.inputs)))
            cli.save_config(cfg)

    # ---- one side -----------------------------------------------------

    @property
    def busy(self) -> bool:
        return self.status in ("waiting", "recording", "splitting", "saving")

    def start(self, tape: str, side: str):
        with self.lock:
            if self.busy:
                raise RuntimeError("Already working on a side.")
            if self.stream is None:
                raise RuntimeError(self.source_error or "No audio input is open.")
            name = cli.side_name(tape, side)
            if (self.work_dir / name / f"{name}.flac").exists():
                raise RuntimeError(f"{name} is already recorded. Delete it from the library first to record it again.")
            self.side, self.status, self.message, self.job = name, "waiting", "Press play on the deck", "record"
            self.elapsed = self.silent_for = 0.0
            self.live_wave, self._wave_acc = [], []
            self.stop_requested = False
            self.rec_q = queue.Queue()
        threading.Thread(target=self._run, args=(name,), daemon=True).start()

    def stop(self):
        self.stop_requested = True

    def _blocks(self):
        while True:
            try:
                yield self.rec_q.get(timeout=0.5)
            except queue.Empty:
                if self.stop_requested:
                    return

    def _progress(self, det):
        if det.state is State.WAITING:
            return
        if self.status == "waiting":
            self.status, self.message = "recording", "Recording"
        self.elapsed = det.recorded_frames / self.settings.samplerate
        self.silent_for = det._silent_run
        self._wave_acc.append(det.level_db)
        if len(self._wave_acc) >= 5:  # one point per half second
            self.live_wave.append(max(self._wave_acc))
            self._wave_acc = []

    def _run(self, name: str):
        side_dir = self.work_dir / name
        try:
            res = record_side(side_dir, name, self.settings, self._blocks(), self._progress, lambda: self.stop_requested)
            self.rec_q = None
            if res is None:
                self.status, self.message = "idle", "Stopped before any sound was heard. Nothing was saved."
                shutil.rmtree(side_dir, ignore_errors=True)
                return
            self.status, self.message = "splitting", "Finding the tracks and saving to Google Drive"
            cli.split_side(side_dir, rerun_hint="the editor")
            check = (side_dir / "CHECK ME.txt").exists()
            ended = {"silence": "the end of the side", "max_length": "the length limit", "stopped": "you pressing Stop"}[res.ended_by]
            self.status = "done"
            self.message = f"Done. Stopped at {ended}." + (" Some splits are guesses, have a look." if check else "")
        except Exception as e:
            traceback.print_exc()
            self.status, self.message = "error", f"Something went wrong: {e}"
        finally:
            self.rec_q = None

    def save_edits(self, name: str, tracks: list[dict]):
        """Re-cut a side with edited markers and titles, in the background."""
        side_dir = self.side_dir(name)
        with self.lock:
            if self.busy:
                raise RuntimeError("Wait for the current side to finish first.")
            self.status, self.side, self.message, self.job = "saving", name, "Re-cutting and saving to Google Drive", "edit"
        threading.Thread(target=self._save, args=(side_dir, name, tracks), daemon=True).start()

    def save_draft(self, name: str, tracks: list[dict] | None):
        """Keep edits to come back to later, without cutting or uploading anything.
        `None` throws the draft away."""
        f = self.side_dir(name) / "draft.json"
        if tracks is None:
            f.unlink(missing_ok=True)
            return
        clean = [{"start": float(t["start"]), "end": float(t["end"]), "skip": bool(t.get("skip")), "title": str(t.get("title", ""))} for t in tracks]
        f.write_text(json.dumps(clean))

    def _save(self, side_dir: Path, name: str, tracks: list[dict]):
        try:
            # Skipped sections are simply left out of the label file.
            items = sorted((float(t["start"]), float(t["end"]), str(t.get("title", "")).strip()) for t in tracks if not t.get("skip"))
            if not items:
                raise ValueError("every section is marked Skip, so there would be nothing to save")
            splitter.write_labels(side_dir / "tracks.txt", [splitter.Track(a, b, title) for a, b, title in items])
            cli.split_side(side_dir, labels=side_dir / "tracks.txt")
            (side_dir / "draft.json").unlink(missing_ok=True)
            self.status, self.message = "done", f"Saved {len(items)} tracks for {name}."
        except Exception as e:
            traceback.print_exc()
            self.status, self.message = "error", f"Couldn't save: {e}"

    # ---- library ------------------------------------------------------

    def side_dir(self, name: str) -> Path:
        if not SIDE_RE.match(name) or "/" in name or ".." in name:
            raise ValueError("bad side name")
        return self.work_dir / name

    def library(self) -> list[dict]:
        log = {}
        dest = cli.drive_folder()
        if dest and (dest / "log.csv").exists():
            with (dest / "log.csv").open(newline="") as f:
                for r in csv.DictReader(f):
                    log[f"{r['tape']} - Side {r['side']}"] = r
        sides = []
        if self.work_dir.exists():
            for d in sorted(self.work_dir.iterdir()):
                master = d / f"{d.name}.flac"
                if not (d.is_dir() and SIDE_RE.match(d.name) and master.exists()):
                    continue
                tracks = sorted((d / "tracks").glob("*.flac")) if (d / "tracks").is_dir() else []
                r = log.get(d.name, {})
                sides.append({
                    "name": d.name,
                    "minutes": round(splitter.sf.info(master).duration / 60, 1),
                    "tracks": len(tracks),
                    "needs_check": (d / "CHECK ME.txt").exists(),
                    "draft": (d / "draft.json").exists(),
                    "saved": r.get("finished", ""),
                    "recorded": time.strftime("%Y-%m-%d %H:%M", time.localtime(master.stat().st_mtime)),
                })
        return sides

    def side_detail(self, name: str) -> dict:
        d = self.side_dir(name)
        master = d / f"{name}.flac"
        info = splitter.sf.info(master)
        tracks = splitter.read_labels(d / "tracks.txt", info.duration) if (d / "tracks.txt").exists() else [splitter.Track(0, info.duration)]
        cache = d / "wave.json"
        if cache.exists() and cache.stat().st_mtime >= master.stat().st_mtime:
            wave = json.loads(cache.read_text())
        else:
            env, _ = splitter.level_envelope(master, 0.25)
            wave = [round(float(x), 1) for x in env]
            cache.write_text(json.dumps(wave))
        notes = (d / "CHECK ME.txt").read_text().split("\n\n")[0] if (d / "CHECK ME.txt").exists() else ""
        saved = [
            {"start": t.start, "end": t.end, "skip": skip, "title": "" if t.title.startswith("Track ") else t.title}
            for t, skip in splitter.fill_gaps(tracks, info.duration)
        ]
        draft = None
        if (d / "draft.json").exists():
            try:
                draft = json.loads((d / "draft.json").read_text())
            except ValueError:
                draft = None
        return {
            "name": name,
            "duration": info.duration,
            "wave_step": 0.25,
            "wave": wave,
            "saved_tracks": saved,  # what was last cut and uploaded
            "tracks": draft or saved,  # what to show: the draft if there is one
            "draft": draft is not None,
            "notes": notes,
        }

    def wave_window(self, name: str, start: float, end: float, step: float = 0.01) -> dict:
        """Fine-grained loudness for a short stretch, for placing a cut precisely."""
        master = self.side_dir(name) / f"{name}.flac"
        info = splitter.sf.info(master)
        start, end = max(0.0, start), min(info.duration, end)
        if end - start > 120:
            raise ValueError("window too long")
        frame = max(1, int(info.samplerate * step))
        with splitter.sf.SoundFile(master) as f:
            f.seek(int(start * info.samplerate))
            audio = f.read(int((end - start) * info.samplerate), dtype="float32", always_2d=True).mean(axis=1)
        n = len(audio) // frame
        rms = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame).astype(np.float64) ** 2, axis=1)) if n else np.zeros(0)
        return {"start": start, "step": step, "wave": [round(db(float(x)), 1) for x in rms]}

    def next_side(self) -> dict:
        """Suggest the next numbered side: B of the latest tape, else A of the one after."""
        names = {s["name"] for s in self.library()}
        nums = [int(m.group(1)) for n in names if (m := re.match(r"^Tape (\d+) - Side ", n))]
        if not nums:
            return {"tape": "1", "side": "A"}
        top = max(nums)
        if cli.side_name(top, "B") not in names:
            return {"tape": str(top), "side": "B"}
        return {"tape": str(top + 1), "side": "A"}

    def state(self) -> dict:
        names = {i: n for i, n, _, _ in list_inputs()}
        return {
            "status": self.status,
            "message": self.message,
            "side": self.side,
            "job": self.job,
            "elapsed": round(self.elapsed, 1),
            "silent_for": round(self.silent_for, 1),
            "end_silence": self.settings.end_silence_seconds,
            "live_wave": self.live_wave[-2400:],
            "meter": self.meter,
            "source": {
                "device": self.device,
                "device_name": names.get(self.device, "Default input"),
                "inputs": list(self.inputs),
                "error": self.source_error,
            },
        }


def make_handler(engine: Engine):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send_json(self, data, code=200):
            body = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def read_json(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path in ("/", "/index.html"):
                    return self.send_file(STATIC / "index.html")
                if url.path == "/api/state":
                    return self.send_json(engine.state())
                if url.path == "/api/devices":
                    return self.send_json([{"id": i, "name": n, "inputs": c} for i, n, c, _ in list_inputs()])
                if url.path == "/api/library":
                    return self.send_json({"sides": engine.library(), "next": engine.next_side(), "drive": str(cli.drive_folder() or "")})
                if url.path == "/api/side":
                    return self.send_json(engine.side_detail(q["name"]))
                if url.path == "/api/wave":
                    return self.send_json(engine.wave_window(q["name"], float(q["start"]), float(q["end"])))
                if url.path == "/api/audio":
                    d = engine.side_dir(q["name"])
                    return self.send_file(d / f"{d.name}.flac", "audio/flac")
            except (KeyError, ValueError, FileNotFoundError) as e:
                return self.send_json({"error": str(e)}, 404)
            self.send_json({"error": "not found"}, 404)

        def do_POST(self):
            url = urlparse(self.path)
            try:
                body = self.read_json()
                if url.path == "/api/source":
                    engine.open_source(body.get("device"), parse_inputs(body.get("inputs", "1,2")))
                    return self.send_json(engine.state())
                if url.path == "/api/start":
                    engine.start(str(body["tape"]).strip(), str(body["side"]).strip())
                    return self.send_json(engine.state())
                if url.path == "/api/stop":
                    engine.stop()
                    return self.send_json(engine.state())
                if url.path == "/api/side/draft":
                    engine.save_draft(body["name"], body.get("tracks"))
                    return self.send_json({"ok": True})
                if url.path == "/api/side/save":
                    engine.save_edits(body["name"], body["tracks"])
                    return self.send_json({"ok": True})
                if url.path == "/api/reveal":
                    d = engine.side_dir(body["name"]) if body.get("name") else cli.drive_folder()
                    subprocess.run(["open", str(d)], check=False)
                    return self.send_json({"ok": True})
                if url.path == "/api/side/delete":
                    d = engine.side_dir(body["name"])
                    trash = engine.work_dir.parent / "deleted"
                    trash.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(d), str(trash / f"{d.name} ({time.strftime('%Y%m%d-%H%M%S')})"))
                    return self.send_json({"ok": True})
            except (RuntimeError, ValueError, KeyError) as e:
                return self.send_json({"error": str(e)}, 400)
            self.send_json({"error": "not found"}, 404)

        def send_file(self, path: Path, ctype: str | None = None):
            """Serve a file, with Range support so audio can be scrubbed."""
            size = path.stat().st_size
            ctype = ctype or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            start, end = 0, size - 1
            rng = self.headers.get("Range")
            m = re.match(r"bytes=(\d*)-(\d*)", rng or "")
            if m:
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else end
                else:
                    start = size - int(m.group(2))
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(200)
            self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype == "text/html" else ""))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with path.open("rb") as f:
                f.seek(start)
                left = end - start + 1
                try:
                    while left > 0:
                        chunk = f.read(min(1 << 20, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass

    return Handler


def open_window(url: str):
    """A browser-style app window (Chrome or Edge app mode), else a normal tab."""
    for app in ("Google Chrome", "Microsoft Edge", "Brave Browser"):
        if Path(f"/Applications/{app}.app").exists():
            subprocess.Popen(["open", "-na", app, "--args", f"--app={url}", "--window-size=1180,860"])
            return
    webbrowser.open(url)


def run(work_dir: Path = cli.DEFAULT_WORK_DIR, port: int = 8765, open_browser: bool = True):
    url = f"http://127.0.0.1:{port}/"
    engine = Engine(work_dir)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(engine))
    except OSError:
        # Already running (e.g. double-clicked twice): just bring up a window.
        if engine.stream is not None:
            engine.stream.close()
        print("Tape Digitizer is already running, opening its window.")
        open_window(url)
        return
    print(f"Tape Digitizer is running at {url}  (close this window or press Ctrl+C to quit)")
    if open_browser:
        threading.Timer(0.5, open_window, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if engine.stream is not None:
            engine.stream.close()
