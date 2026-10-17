"""Save a split side to Google Drive as tagged MP3s, and log it.

Google Drive for desktop keeps a synced folder on the Mac, so writing files
there is all it takes for them to upload.

    <Drive>/Music/Tapes/Tape 001/Side A/01.mp3
    <Drive>/Music/Tapes/Masters/Tape 001 - Side A.flac   (the whole side, lossless)
    <Drive>/Music/Tapes/log.csv
"""

from __future__ import annotations

import csv
import shutil
import time
from pathlib import Path

import soundfile as sf

LOG_FIELDS = ["tape", "side", "finished", "minutes", "tracks", "split_by", "needs_check", "folder"]


def find_drive_folder() -> Path | None:
    """Music/Tapes inside the Google Drive for desktop folder, if Drive is installed."""
    roots = sorted((Path.home() / "Library" / "CloudStorage").glob("GoogleDrive-*/My Drive"))
    return roots[0] / "Music" / "Tapes" if roots else None


def parse_side_name(name: str) -> tuple[str, str]:
    """'Tape 001 - Side A' -> ('Tape 001', 'A')."""
    tape, _, side = name.rpartition(" - Side ")
    return (tape, side) if tape else (name, "")


def copy_master(side_dir: Path, dest_root: Path) -> Path | None:
    """Copy the full-side FLAC to Masters/, skipping it if an identical copy is there."""
    src = side_dir / f"{side_dir.name}.flac"
    if not src.exists():
        return None
    dest = dest_root / "Masters" / src.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not (dest.exists() and dest.stat().st_size == src.stat().st_size and dest.stat().st_mtime >= src.stat().st_mtime):
        shutil.copy2(src, dest)
    return dest


def export_side(side_dir: Path, dest_root: Path, split_by: str = "", needs_check: bool = False, masters: bool = True) -> Path:
    """Convert the side's tracks to 320 kbps MP3 in `dest_root/<tape>/Side <x>/`.

    Earlier exports of the same side are replaced, so re-splitting a side and
    exporting again leaves only the new tracks.
    """
    tape, side = parse_side_name(side_dir.name)
    dest = dest_root / tape / (f"Side {side}" if side else "")
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob("*.mp3"):
        old.unlink()

    tracks = sorted((side_dir / "tracks").glob("*.flac"))
    total = 0.0
    for n, src in enumerate(tracks, 1):
        audio, rate = sf.read(src, dtype="float32", always_2d=True)
        total += len(audio) / rate
        title = src.stem.split(" - ", 1)[1] if " - " in src.stem else f"Track {n:02d}"
        with sf.SoundFile(
            dest / f"{src.stem}.mp3", "w", rate, audio.shape[1],
            format="MP3", subtype="MPEG_LAYER_III", compression_level=0.0, bitrate_mode="CONSTANT",
        ) as f:
            f.title = title
            f.album = f"{tape} Side {side}".strip()
            f.artist = "Tapes"
            f.tracknumber = str(n)
            f.write(audio)

    if masters:
        copy_master(side_dir, dest_root)

    write_log(dest_root / "log.csv", {
        "tape": tape,
        "side": side,
        "finished": time.strftime("%Y-%m-%d %H:%M"),
        "minutes": f"{total / 60:.1f}",
        "tracks": len(tracks),
        "split_by": split_by,
        "needs_check": "yes" if needs_check else "",
        "folder": str(dest.relative_to(dest_root)),
    })
    return dest


def remove_side(dest_root: Path, name: str) -> dict | None:
    """Take a side out of Drive: its MP3s, its master and its log row.

    Returns the side's old log row, if it had one.
    """
    tape, side = parse_side_name(name)
    folder = dest_root / tape / (f"Side {side}" if side else "")
    if folder.is_dir():
        for old in folder.glob("*.mp3"):
            old.unlink()
        for d in (folder, folder.parent):
            if d != dest_root and d.is_dir() and not any(d.iterdir()):
                d.rmdir()
    (dest_root / "Masters" / f"{name}.flac").unlink(missing_ok=True)
    path, row = dest_root / "log.csv", None
    if path.exists():
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        keep = [r for r in rows if (r["tape"], r["side"]) != (tape, side)]
        row = next((r for r in rows if (r["tape"], r["side"]) == (tape, side)), None)
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            w.writeheader()
            w.writerows(keep)
    return row


def write_log(path: Path, row: dict) -> None:
    """One row per side; exporting a side again updates its row."""
    rows = []
    if path.exists():
        with path.open(newline="") as f:
            rows = [r for r in csv.DictReader(f) if (r["tape"], r["side"]) != (row["tape"], row["side"])]
    rows.append({k: str(v) for k, v in row.items()})
    rows.sort(key=lambda r: (r["tape"], r["side"]))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        w.writeheader()
        w.writerows(rows)
