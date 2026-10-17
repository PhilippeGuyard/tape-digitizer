"""Command line for the tape digitizer.

    python -m tapedigitizer app                  open the Tape Digitizer window
    python -m tapedigitizer devices              list audio devices and their inputs
    python -m tapedigitizer choose               pick the device and inputs (remembered)
    python -m tapedigitizer levels               live level meter
    python -m tapedigitizer calibrate            measure tape hiss, suggest thresholds
    python -m tapedigitizer record 42 A          record tape 42, side A, then split it
    python -m tapedigitizer split 42 A           (re)split a recorded side into tracks
    python -m tapedigitizer export 42 A          (re)save a side's tracks to Google Drive
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import exporter, recognize, splitter
from .recorder import RecorderSettings, State, find_device, list_inputs, live_blocks, parse_inputs, record_side, rms_db

DEFAULT_WORK_DIR = Path.home() / "TapeDigitizer" / "work"
CONFIG_FILE = Path.home() / "TapeDigitizer" / "config.json"


def load_config() -> dict:
    try:
        return json.loads(CONFIG_FILE.read_text())
    except (OSError, ValueError):
        return {}


def save_config(cfg: dict) -> None:
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


def meter(db: float, width: int = 40) -> str:
    filled = int(max(0.0, min(1.0, (db + 70) / 70)) * width)
    return "#" * filled + "-" * (width - filled)


def resolve_source(args):
    """Device index and input numbers: command line first, then saved choice, then Focusrite 1+2."""
    cfg = load_config()
    dev_arg = args.device
    if dev_arg is None and cfg.get("device_name"):
        # Device numbers shift when things are plugged in differently; match the saved name.
        dev_arg = next((i for i, n, _, _ in list_inputs() if n == cfg["device_name"]), cfg.get("device"))
    elif dev_arg is None:
        dev_arg = cfg.get("device")
    if dev_arg is None:
        dev = find_device()
        if dev is None:
            print("No Focusrite found, using the default input. Run './tape choose' to pick one.")
    elif str(dev_arg).isdigit():
        dev = int(dev_arg)
    else:
        dev = find_device(str(dev_arg))
        if dev is None:
            sys.exit(f"No input device matching '{dev_arg}'. Run './tape devices' to see them.")
    inputs = parse_inputs(args.inputs or cfg.get("inputs", "1,2"))
    return dev, inputs


def describe_source(dev, inputs) -> str:
    names = {i: n for i, n, _, _ in list_inputs()}
    label = "input " + " + ".join(map(str, inputs))
    kind = "stereo" if len(inputs) == 2 else "mono"
    return f"{names.get(dev, 'default input')}, {label} ({kind})"


def cmd_devices(args):
    for i, name, n, rate in list_inputs():
        print(f"{i:3d}  {name}  ({n} inputs, {rate} Hz)")


def cmd_choose(args):
    devs = list_inputs()
    if not devs:
        sys.exit("No audio inputs found. Is the audio interface plugged in?")
    for i, name, n, rate in devs:
        print(f"{i:3d}  {name}  ({n} inputs)")
    default = find_device()
    raw = input(f"Device number [{default if default is not None else devs[0][0]}]: ").strip()
    dev = int(raw) if raw else (default if default is not None else devs[0][0])
    n = next((c for i, _, c, _ in devs if i == dev), None)
    if n is None:
        sys.exit(f"{dev} is not one of the devices listed.")
    suggestion = "1,2" if n >= 2 else "1"
    raw = input(f"Inputs to record, e.g. 1,2 for stereo or 1 for mono (this device has {n}) [{suggestion}]: ").strip()
    inputs = parse_inputs(raw or suggestion)
    if max(inputs) > n:
        sys.exit(f"This device only has {n} inputs.")
    cfg = load_config()
    cfg.update(device=dev, device_name=next(nm for i, nm, _, _ in devs if i == dev), inputs=",".join(map(str, inputs)))
    save_config(cfg)
    print(f"Saved: {describe_source(dev, inputs)}. Check it with './tape levels'.")


def cmd_levels(args):
    s = RecorderSettings()
    dev, inputs = resolve_source(args)
    print(f"{describe_source(dev, inputs)}. Press Ctrl+C to stop.")
    try:
        for block in live_blocks(s, dev, inputs):
            parts = []
            clip = False
            for k, inp in enumerate(inputs):
                ch = block[:, k]
                db = rms_db(ch)
                peak = 20 * np.log10(max(float(np.max(np.abs(ch))), 1e-6))
                clip |= peak > -1
                parts.append(f"in{inp} {meter(db, 24)} {db:6.1f} dB")
            warn = "  CLIPPING" if clip else ""
            print("\r" + "   ".join(parts) + warn + "   ", end="", flush=True)
    except KeyboardInterrupt:
        print()


def cmd_calibrate(args):
    s = RecorderSettings()
    print(f"Play a blank stretch of tape (or the leader) for {args.seconds} seconds...")
    dev, inputs = resolve_source(args)
    levels = []
    start = time.time()
    for block in live_blocks(s, dev, inputs):
        levels.append(rms_db(block))
        if time.time() - start > args.seconds:
            break
    floor = float(np.percentile(levels, 90))
    print(f"Tape hiss is about {floor:.1f} dB.")
    print(f"Suggested: --silence-db {floor + 6:.0f} --start-db {floor + 15:.0f}")


def side_name(tape, side) -> str:
    """'12', 'A' -> 'Tape 012 - Side A'; a named tape keeps its name (minus characters
    that can't go in a file name)."""
    tape = str(tape).strip()
    tape = f"Tape {int(tape):03d}" if tape.isdigit() else "".join(c for c in tape if c not in '/\\:*?"<>|').strip()
    if not tape:
        raise ValueError("give the tape a number or a name")
    return f"{tape} - Side {side.strip().upper()}"


def drive_folder() -> Path | None:
    cfg = load_config()
    return Path(cfg["drive_folder"]).expanduser() if cfg.get("drive_folder") else exporter.find_drive_folder()


def export(side_dir: Path, split_by: str = "", needs_check: bool = False) -> int:
    dest_root = drive_folder()
    if dest_root is None:
        print("Google Drive folder not found, tracks kept only in the work folder.")
        print('Set it with: ./tape set-drive "/path/to/Music/Tapes"')
        return 1
    masters = load_config().get("masters_to_drive", True)
    dest = exporter.export_side(side_dir, dest_root, split_by, needs_check, masters)
    print(f"Saved MP3s to Google Drive: {dest}")
    if masters:
        print(f"Saved the full-side master to: {dest_root / 'Masters'}")
    return 0


def split_side(side_dir: Path, labels: Path | None = None, rerun_hint: str = "", recognize_songs: bool = False, upload: bool = True) -> int:
    master = side_dir / f"{side_dir.name}.flac"
    if not master.exists():
        print(f"No recording found at {master}")
        return 1
    label_file = side_dir / "tracks.txt"
    needs_check = False
    if labels:
        tracks = splitter.read_labels(labels, total=splitter.sf.info(master).duration)
        method, notes = "your labels", []
    else:
        print("Finding tracks...")
        res = splitter.find_tracks(master)
        tracks, method, notes = res.tracks, res.method, res.notes
        token = recognize.load_token() if recognize_songs else None
        if token:
            print("Identifying songs with AudD...")
            try:
                tracks, matches = recognize.refine(master, res, token)
                hits = sum(1 for m in matches if m.key)
                method += f", {hits} of {len(matches)} clips recognized"
                if res.gapless and hits:
                    notes = [n + " (adjusted using song recognition)" for n in notes]
            except (OSError, RuntimeError) as e:
                print(f"  song recognition failed, keeping the splits as found: {e}")
        splitter.write_labels(label_file, tracks)
        needs_check = not res.confident
        if not res.confident:
            (side_dir / "CHECK ME.txt").write_text(
                "The track splits on this side are uncertain:\n- " + "\n- ".join(notes) + "\n\n"
                "To fix: open the .flac in Audacity, File > Import > Labels > tracks.txt, drag the\n"
                "labels (or click to add points), File > Export > Export Labels over tracks.txt,\n"
                f"then run: {rerun_hint or './tape split <tape> <side>'} --edited\n"
            )
    files = splitter.cut_tracks(master, tracks, side_dir / "tracks")
    print(f"{len(files)} tracks ({method}):")
    for f, t in zip(files, tracks):
        m, sec = divmod(int(t.duration), 60)
        print(f"  {m:2d}:{sec:02d}  {f.name}")
    for n in notes:
        print(f"  note: {n}")
    if labels is None and notes:
        print(f"Check the markers in Audacity (see 'CHECK ME.txt' in {side_dir}).")
    elif labels is not None:
        (side_dir / "CHECK ME.txt").unlink(missing_ok=True)
    return export(side_dir, method.split(",")[0], needs_check) if upload else 0


def cmd_split(args):
    side_dir = Path(args.out) / side_name(args.tape, args.side)
    labels = Path(args.labels) if args.labels else (side_dir / "tracks.txt" if args.edited else None)
    return split_side(side_dir, labels, f"./tape split {args.tape} {args.side}", args.recognize, not args.no_upload)


def cmd_export(args):
    side_dir = Path(args.out) / side_name(args.tape, args.side)
    if not (side_dir / "tracks").is_dir():
        sys.exit(f"No tracks found in {side_dir}. Run './tape split {args.tape} {args.side}' first.")
    return export(side_dir, "", (side_dir / "CHECK ME.txt").exists())


def cmd_app(args):
    from .app import run

    run(Path(args.out), args.port, not args.no_browser)


def cmd_set_drive(args):
    path = Path(args.folder).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    cfg["drive_folder"] = str(path)
    save_config(cfg)
    print(f"Tracks will be saved to {path}")


def cmd_identify(args):
    """Look up one clip and show AudD's full reply, to check recognition works."""
    import os

    side_dir = Path(args.out) / side_name(args.tape, args.side)
    master = side_dir / f"{side_dir.name}.flac"
    token = recognize.load_token()
    if not token:
        sys.exit(f"No AudD token found in {recognize.TOKEN_FILE}")
    m, sec = (args.at.split(":") + ["0"])[:2] if ":" in args.at else ("0", args.at)
    at = int(m) * 60 + float(sec)
    os.environ["TAPE_DEBUG"] = "1"
    speeds = [1.0] + ([0.97, 1.03, 0.94, 1.06] if args.speeds else [])
    for i, speed in enumerate(speeds):
        wav = recognize.clip_wav(master, at, args.seconds, speed=speed)
        if i == 0:
            clip = side_dir / f"clip at {int(at // 60)}m{int(at % 60):02d}s.wav"
            clip.write_bytes(wav)
            print(f"Sent {args.seconds:.0f} s from {int(at // 60)}:{int(at % 60):02d} (saved as {clip.name} so you can listen to it).")
        else:
            print(f"Trying at {speed * 100:.0f}% speed...")
        try:
            res = recognize.audd_lookup(wav, token)
        except (OSError, RuntimeError) as e:
            print(f"Failed: {e}")
            return 1
        print(f"Result: {res.get('artist')} - {res.get('title')}" if res else "Result: no match")
        if res:
            break
    return 0


def cmd_import(args):
    """Bring in an existing recording (e.g. from Ableton): one stereo file, or left and right mono files."""
    name = side_name(args.tape, args.side)
    side_dir = Path(args.out) / name
    side_dir.mkdir(parents=True, exist_ok=True)
    parts = [splitter.sf.read(f, dtype="float32", always_2d=True) for f in args.files]
    rate = parts[0][1]
    if any(r != rate for _, r in parts):
        sys.exit("The files have different sample rates.")
    n = min(len(a) for a, _ in parts)
    audio = np.concatenate([a[:n] for a, _ in parts], axis=1)
    if audio.shape[1] > 2:
        sys.exit("Give one stereo file or two mono files.")
    splitter.sf.write(side_dir / f"{name}.flac", audio, rate, subtype="PCM_24", format="FLAC")
    print(f"Imported {name} ({n / rate / 60:.1f} min, {'stereo' if audio.shape[1] == 2 else 'mono'}).")
    return split_side(side_dir, rerun_hint=f"./tape split {args.tape} {args.side}", recognize_songs=args.recognize, upload=not args.no_upload)


def cmd_record(args):
    s = RecorderSettings(
        start_threshold_db=args.start_db,
        silence_threshold_db=args.silence_db,
        end_silence_seconds=args.end_silence,
    )
    name = side_name(args.tape, args.side)
    out_dir = Path(args.out) / name
    dev, inputs = resolve_source(args)
    print(f"Recording from {describe_source(dev, inputs)}")
    print(f"Ready: {name}. Press play on the deck. (Ctrl+C to stop early.)")

    def progress(det):
        if det.state is State.WAITING:
            line = f"waiting for sound  {meter(det.level_db)} {det.level_db:6.1f} dB"
        else:
            t = det.recorded_frames / s.samplerate
            silent = det._silent_run
            tail = f"  silent {silent:4.0f}s" if silent >= 2 else ""
            line = f"recording {int(t // 60):02d}:{int(t % 60):02d}  {meter(det.level_db)} {det.level_db:6.1f} dB{tail}"
        print(f"\r{line}   ", end="", flush=True)

    try:
        result = record_side(out_dir, name, s, live_blocks(s, dev, inputs), progress)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 1
    print()
    if result is None:
        print("No sound was detected, nothing saved.")
        return 1
    m, sec = divmod(int(result.duration_seconds), 60)
    print(f"Saved {out_dir / result.master}  ({m}:{sec:02d}, peak {result.peak_db} dB, ended by {result.ended_by})")
    if result.peak_db > -1:
        print("Warning: the recording clipped. Turn the input gain down a little.")
    if args.no_split:
        return 0
    return split_side(out_dir, rerun_hint=f"./tape split {args.tape} {args.side}", upload=not args.no_upload)


def main(argv=None):
    p = argparse.ArgumentParser(prog="tapedigitizer", description="Digitize cassette tapes hands-free.")
    src = argparse.ArgumentParser(add_help=False)
    src.add_argument("--device", help="device number or part of its name (default: your saved choice, else Focusrite)")
    src.add_argument("--inputs", help="inputs to record: '1,2' stereo, '1' mono, '3-4' (default: saved choice, else 1,2)")
    sub = p.add_subparsers(dest="cmd", required=True)

    ap = sub.add_parser("app", help="open the Tape Digitizer window")
    ap.add_argument("--out", default=str(DEFAULT_WORK_DIR))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.set_defaults(func=cmd_app)

    sub.add_parser("devices", help="list audio devices and their inputs").set_defaults(func=cmd_devices)
    sub.add_parser("choose", help="pick the device and inputs to record from").set_defaults(func=cmd_choose)
    sub.add_parser("levels", parents=[src], help="live level meter").set_defaults(func=cmd_levels)

    c = sub.add_parser("calibrate", parents=[src], help="measure tape hiss and suggest thresholds")
    c.add_argument("--seconds", type=float, default=10)
    c.set_defaults(func=cmd_calibrate)

    r = sub.add_parser("record", parents=[src], help="record one side")
    r.add_argument("tape", help="tape number or name")
    r.add_argument("side", help="A or B")
    r.add_argument("--out", default=str(DEFAULT_WORK_DIR))
    d = RecorderSettings()
    r.add_argument("--start-db", type=float, default=d.start_threshold_db)
    r.add_argument("--silence-db", type=float, default=d.silence_threshold_db)
    r.add_argument("--end-silence", type=float, default=d.end_silence_seconds, help="seconds of silence that end a side")
    r.add_argument("--no-split", action="store_true", help="only record, don't split into tracks")
    r.add_argument("--no-upload", action="store_true", help="don't save MP3s to Google Drive")
    r.set_defaults(func=cmd_record)

    sp = sub.add_parser("split", help="split a recorded side into tracks")
    sp.add_argument("tape")
    sp.add_argument("side")
    sp.add_argument("--out", default=str(DEFAULT_WORK_DIR))
    sp.add_argument("--edited", action="store_true", help="cut using tracks.txt as you edited it in Audacity")
    sp.add_argument("--labels", help="cut using another Audacity label file")
    sp.add_argument("--recognize", action="store_true", help="also look songs up with AudD (off by default)")
    sp.add_argument("--no-upload", action="store_true", help="don't save MP3s to Google Drive")
    sp.set_defaults(func=cmd_split)

    ex = sub.add_parser("export", help="save a side's tracks to Google Drive as MP3s")
    ex.add_argument("tape")
    ex.add_argument("side")
    ex.add_argument("--out", default=str(DEFAULT_WORK_DIR))
    ex.set_defaults(func=cmd_export)

    sd_ = sub.add_parser("set-drive", help="choose a different folder for the MP3s")
    sd_.add_argument("folder")
    sd_.set_defaults(func=cmd_set_drive)

    idf = sub.add_parser("identify", help="test song recognition on one clip of a recorded side")
    idf.add_argument("tape")
    idf.add_argument("side")
    idf.add_argument("at", help="where in the side, e.g. 1:30")
    idf.add_argument("--seconds", type=float, default=12)
    idf.add_argument("--speeds", action="store_true", help="also try 3% and 6% faster and slower, for off-speed decks")
    idf.add_argument("--out", default=str(DEFAULT_WORK_DIR))
    idf.set_defaults(func=cmd_identify)

    im = sub.add_parser("import", help="bring in an existing recording, e.g. from Ableton, and split it")
    im.add_argument("tape")
    im.add_argument("side")
    im.add_argument("files", nargs="+", help="one stereo file, or the left then right mono files")
    im.add_argument("--out", default=str(DEFAULT_WORK_DIR))
    im.add_argument("--recognize", action="store_true", help="also look songs up with AudD (off by default)")
    im.add_argument("--no-upload", action="store_true", help="don't save MP3s to Google Drive")
    im.set_defaults(func=cmd_import)

    args = p.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
