"""Find where the tracks are in a recorded side, and cut it up.

Three ways of finding boundaries, tried in order:

1. Silence: the quiet gaps between songs on a normal tape.
2. Change detection: for seamless mixes, points where the music changes
   character (timbre and harmony), using librosa.
3. Song recognition (optional, see recognize.py) refines or names them.

Boundaries are saved as an Audacity label file (`tracks.txt`) so they can be
checked and dragged in Audacity, then re-imported with `./tape split --labels`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass
class SplitSettings:
    frame_seconds: float = 0.05
    # Gaps on home-made tapes can be very short (a pause-button dip of half a
    # second), so anything this long can count...
    min_gap_seconds: float = 0.3
    # ...as long as it drops this far below the music around it.
    gap_depth_db: float = 18.0
    # How much music around a dip is used to judge "the music around it".
    context_seconds: float = 20.0
    # Tracks shorter than this are merged into a neighbour (quiet intros, pauses).
    min_track_seconds: float = 40.0
    # Anything longer than this is probably several songs run together: it is
    # searched for changes in the music too.
    long_track_seconds: float = 7 * 60
    # Silence at least this long in the middle of a side is left out of the tracks.
    long_silence_seconds: float = 8.0
    # How much of the lead-in and run-out is kept around the music.
    trim_margin_start: float = 0.5
    trim_margin_end: float = 1.0
    # Change detection: shortest believable track in a mix.
    mix_min_track_seconds: float = 120.0


@dataclass
class Track:
    start: float
    end: float
    title: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class SplitResult:
    tracks: list[Track]
    method: str  # "silence", "changes" or "none"
    confident: bool  # False means the markers deserve a look in Audacity
    notes: list[str] = field(default_factory=list)
    # Stretches with no gaps that were split by change detection, and the
    # change points found in them (recognition can refine these).
    gapless: list[tuple[float, float]] = field(default_factory=list)
    change_points: list[float] = field(default_factory=list)


def level_envelope(path: Path, frame_seconds: float) -> tuple[np.ndarray, int]:
    """Loudness (dBFS) per short frame, read in chunks so long sides fit in memory."""
    info = sf.info(path)
    frame = max(1, int(info.samplerate * frame_seconds))
    levels = []
    for chunk in sf.blocks(str(path), blocksize=frame * 200, dtype="float32", always_2d=True):
        mono = chunk.mean(axis=1)
        n = len(mono) // frame
        if n:
            f = mono[: n * frame].reshape(n, frame)
            levels.append(np.sqrt(np.mean(f.astype(np.float64) ** 2, axis=1)))
        if len(mono) % frame:
            levels.append(np.array([np.sqrt(np.mean(mono[n * frame :].astype(np.float64) ** 2))]))
    env = np.concatenate(levels) if levels else np.zeros(0)
    return 20 * np.log10(np.maximum(env, 1e-6)), info.samplerate


def local_level(db: np.ndarray, frame_seconds: float, window_seconds: float) -> np.ndarray:
    """Typical loudness around each frame: median over a sliding window."""
    per_sec = max(1, int(1 / frame_seconds))
    n_sec = int(np.ceil(len(db) / per_sec))
    padded = np.pad(db, (0, n_sec * per_sec - len(db)), mode="edge")
    seconds = np.median(padded.reshape(n_sec, per_sec), axis=1)
    half = max(1, int(window_seconds / 2))
    med = np.array([np.median(seconds[max(0, i - half) : i + half + 1]) for i in range(n_sec)])
    return np.repeat(med, per_sec)[: len(db)]


def find_gaps(db: np.ndarray, s: SplitSettings) -> list[tuple[float, float]]:
    """(start, end) in seconds of every dip deep and long enough to be a gap.

    A gap is judged against the music around it rather than one fixed level,
    so quiet songs and noisy tapes both work.
    """
    if len(db) == 0:
        return []
    # Light smoothing so a single click inside a gap doesn't break it.
    k = max(1, int(0.2 / s.frame_seconds))
    smooth = np.convolve(db, np.ones(k) / k, mode="same")
    # Quiet relative to the music around it; or, for long silences that fill
    # that whole window, far below the side's typical music level.
    quiet = (smooth < local_level(smooth, s.frame_seconds, s.context_seconds) - s.gap_depth_db) | (
        smooth < np.percentile(smooth, 60) - 30
    )

    gaps = []
    min_frames = max(1, int(round(s.min_gap_seconds / s.frame_seconds)))
    i, n = 0, len(quiet)
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if j - i >= min_frames and i > 0 and j < n:  # ignore lead-in and run-out
                gaps.append((i * s.frame_seconds, j * s.frame_seconds))
            i = j
        else:
            i += 1
    return gaps


def music_bounds(db: np.ndarray, s: SplitSettings) -> tuple[float, float]:
    """Where the music starts and ends, ignoring leader and run-out silence."""
    loud = np.nonzero(db > np.percentile(db, 50) - 25)[0]
    if len(loud) == 0:
        return 0.0, len(db) * s.frame_seconds
    return loud[0] * s.frame_seconds, (loud[-1] + 1) * s.frame_seconds


def tracks_from_boundaries(bounds: list[float], total: float, min_len: float) -> list[Track]:
    """Turn cut points into tracks, merging any that come out too short."""
    edges = [0.0] + sorted(b for b in bounds if 0 < b < total) + [total]
    tracks = [Track(a, b) for a, b in zip(edges, edges[1:])]
    merged: list[Track] = []
    for t in tracks:
        if merged and (t.duration < min_len or merged[-1].duration < min_len):
            merged[-1] = Track(merged[-1].start, t.end)
        else:
            merged.append(t)
    return merged


def split_by_silence(path: Path, s: SplitSettings) -> tuple[list[Track], float]:
    db, _ = level_envelope(path, s.frame_seconds)
    total = len(db) * s.frame_seconds
    start, end = music_bounds(db, s)
    # Cut in the middle of each gap so neither track loses its fade, and
    # ignore anything in the leader or run-out.
    bounds = [(a + b) / 2 for a, b in find_gaps(db, s) if start < a and b < end]
    return tracks_from_boundaries(bounds, total, s.min_track_seconds), total


def find_changes(path: Path, s: SplitSettings, sr: int = 11025, offset: float = 0.0, duration: float | None = None) -> list[float]:
    """Likely track changes in a seamless mix, in seconds.

    Builds a self-similarity matrix of timbre (MFCC) and harmony (chroma) and
    slides a checkerboard kernel along its diagonal (Foote's novelty method):
    peaks are moments where what comes after sounds unlike what came before.
    """
    import librosa
    import scipy.signal

    y, sr = librosa.load(str(path), sr=sr, mono=True, offset=offset, duration=duration)
    hop = 2048  # about 0.19 s
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    # Average over ~2 s windows: we care about sections, not notes.
    win = max(1, int(2.0 * sr / hop))
    feats = np.vstack([librosa.util.normalize(mfcc, axis=1), chroma])
    feats = librosa.util.sync(feats, np.arange(0, feats.shape[1], win), aggregate=np.mean)
    step = win * hop / sr
    feats = librosa.util.normalize(feats, axis=0)
    ssm = feats.T @ feats

    half = max(2, int(30 / step))  # compare 30 s before vs 30 s after
    g = scipy.signal.windows.gaussian(2 * half, std=half / 2)
    kernel = np.outer(g, g) * np.kron([[1, -1], [-1, 1]], np.ones((half, half)))
    n = ssm.shape[0]
    padded = np.pad(ssm, half, mode="edge")
    novelty = np.array([np.sum(padded[i : i + 2 * half, i : i + 2 * half] * kernel) for i in range(n)])
    novelty = np.maximum(novelty, 0)
    if novelty.max() <= 0:
        return []
    novelty /= novelty.max()

    peaks, _ = scipy.signal.find_peaks(
        novelty,
        distance=max(1, int(s.mix_min_track_seconds / step)),
        prominence=0.15,
    )
    # Ignore peaks too close to either end: they are the edges, not changes.
    edge = s.mix_min_track_seconds / 2
    length = len(y) / sr
    return [offset + float(p * step) for p in peaks if edge < p * step < length - edge]


def find_tracks(path: Path, s: SplitSettings | None = None) -> SplitResult:
    """Find the tracks, leaving out the lead-in, run-out and long silences."""
    s = s or SplitSettings()
    res = _find_tracks(path, s)
    res.tracks = subtract(res.tracks, skip_regions(path, s))
    return res


def skip_regions(path: Path, s: SplitSettings) -> list[tuple[float, float]]:
    """Stretches not worth keeping: leader before the music, run-out after it,
    and long silences in between."""
    db, _ = level_envelope(path, s.frame_seconds)
    total = len(db) * s.frame_seconds
    start, end = music_bounds(db, s)
    skips = []
    if start - s.trim_margin_start > 0.5:
        skips.append((0.0, start - s.trim_margin_start))
    for a, b in find_gaps(db, s):
        if b - a >= s.long_silence_seconds and start < a and b < end:
            skips.append((a + 0.5, b - 0.5))
    if total - (end + s.trim_margin_end) > 0.5:
        skips.append((end + s.trim_margin_end, total))
    return skips


def subtract(tracks: list[Track], skips: list[tuple[float, float]], min_len: float = 1.0) -> list[Track]:
    """Cut the skipped stretches out of the tracks (tracks may then have gaps between them)."""
    out = []
    for t in tracks:
        pieces = [(t.start, t.end)]
        for a, b in skips:
            nxt = []
            for x, y in pieces:
                if b <= x or a >= y:
                    nxt.append((x, y))
                    continue
                if a > x:
                    nxt.append((x, a))
                if b < y:
                    nxt.append((b, y))
            pieces = nxt
        out += [Track(x, y, t.title) for x, y in pieces if y - x >= min_len]
    return out


def fill_gaps(tracks: list[Track], total: float) -> list[tuple[Track, bool]]:
    """The whole side as (track, skipped) pairs: tracks, plus the stretches between them."""
    out, pos = [], 0.0
    for t in sorted(tracks, key=lambda t: t.start):
        if t.start - pos > 0.05:
            out.append((Track(pos, t.start), True))
        out.append((t, False))
        pos = max(pos, t.end)
    if total - pos > 0.05:
        out.append((Track(pos, total), True))
    return out


def _find_tracks(path: Path, s: SplitSettings) -> SplitResult:
    """Gaps first; then any stretch too long to be one song is searched for changes.

    This handles normal tapes, seamless mixes, and tapes that are a bit of both.
    """
    tracks, total = split_by_silence(path, s)
    long = [t for t in tracks if t.duration > s.long_track_seconds]
    if not long:
        return SplitResult(tracks, "silence", confident=True)

    cuts = [t.end for t in tracks[:-1]]
    changes: list[float] = []
    try:
        for t in long:
            changes += find_changes(path, s, offset=t.start, duration=t.duration)
    except ImportError:
        return SplitResult(tracks, "silence", False, ["some stretches have no gaps and librosa is not installed"])

    found = len(changes)
    cuts += changes
    gaps = len(tracks) - 1
    method = "changes" if gaps == 0 else "silence+changes"
    min_len = s.mix_min_track_seconds if gaps == 0 else s.min_track_seconds
    tracks = tracks_from_boundaries(cuts, total, min_len)
    mins = ", ".join(f"{t.start / 60:.0f} to {t.end / 60:.0f} min" for t in long)
    notes = [f"no gaps between {mins}: {found} cut(s) there are estimates, check them in Audacity"]
    return SplitResult(tracks, method, False, notes, [(t.start, t.end) for t in long], changes)


# Audacity label files: one line per track, "start<TAB>end<TAB>title".

def write_labels(path: Path, tracks: list[Track]) -> None:
    lines = [f"{t.start:.3f}\t{t.end:.3f}\t{t.title or f'Track {i + 1:02d}'}" for i, t in enumerate(tracks)]
    path.write_text("\n".join(lines) + "\n")


def read_labels(path: Path, total: float | None = None) -> list[Track]:
    """Read an Audacity label file.

    Region labels give exact tracks. Point labels (start == end, what you get
    by clicking once in Audacity) are treated as cut points.
    """
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 2 or parts[0].startswith("\\"):  # skip Audacity's frequency lines
            continue
        rows.append((float(parts[0]), float(parts[1]), parts[2].strip() if len(parts) > 2 else ""))
    rows.sort()
    if rows and all(abs(a - b) < 1e-6 for a, b, _ in rows):
        if total is None:
            raise ValueError("point labels need the side's length")
        tracks = tracks_from_boundaries([a for a, _, _ in rows], total, 0)
        return tracks
    return [Track(a, b, title) for a, b, title in rows]


def cut_tracks(master: Path, tracks: list[Track], out_dir: Path) -> list[Path]:
    """Write each track as its own FLAC, with short fades so cuts don't click."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.flac"):
        old.unlink()
    info = sf.info(master)
    fade = int(0.01 * info.samplerate)
    paths = []
    with sf.SoundFile(master) as src:
        for i, t in enumerate(tracks, 1):
            a, b = int(t.start * info.samplerate), int(t.end * info.samplerate)
            src.seek(a)
            audio = src.read(b - a, dtype="float32", always_2d=True)
            if len(audio) > 2 * fade:
                ramp = np.linspace(0, 1, fade, dtype="float32")[:, None]
                audio[:fade] *= ramp
                audio[-fade:] *= ramp[::-1]
            name = f"{i:02d}" + (f" - {safe_name(t.title)}" if t.title and not t.title.startswith("Track ") else "")
            p = out_dir / f"{name}.flac"
            sf.write(p, audio, info.samplerate, subtype="PCM_24")
            paths.append(p)
    return paths


def safe_name(text: str) -> str:
    return "".join(c for c in text if c not in '/\\:*?"<>|').strip()[:120]
