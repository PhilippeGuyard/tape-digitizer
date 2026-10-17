"""Song recognition with AudD (audd.io), to name tracks and fix splits.

A short clip is sent from each place we want identified. On a normal tape
that's one clip per track; on a mix it's a clip every `mix_every_seconds`.
Where consecutive clips are different songs, there's a track change between
them.

The API token is read from the AUDD_API_TOKEN environment variable, or from
`audd_token.txt` next to the program, which is kept out of version control.
"""

from __future__ import annotations

import io
import json
import os
import time
import urllib.parse
import urllib.request
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .splitter import Track

API_URL = "https://api.audd.io/"
TOKEN_FILE = Path(__file__).resolve().parent.parent / "audd_token.txt"


def load_token() -> str | None:
    tok = os.environ.get("AUDD_API_TOKEN")
    if tok:
        return tok.strip()
    try:
        return TOKEN_FILE.read_text().strip() or None
    except OSError:
        return None


@dataclass
class Match:
    at: float  # seconds into the side where the clip was taken
    artist: str = ""
    title: str = ""
    album: str = ""

    @property
    def key(self) -> str | None:
        return f"{self.artist.lower()}|{self.title.lower()}" if self.title else None

    @property
    def label(self) -> str:
        return f"{self.artist} - {self.title}" if self.artist else self.title


def clip_wav(master: Path, start: float, seconds: float = 12.0, rate: int = 16000, speed: float = 1.0) -> bytes:
    """A short mono WAV cut from the master, small enough to upload quickly.

    `speed` plays the clip faster or slower (1.03 = 3% faster), to make up for
    a deck that runs off-speed, which can stop songs being recognized.
    """
    info = sf.info(master)
    with sf.SoundFile(master) as f:
        f.seek(int(max(0, start) * info.samplerate))
        audio = f.read(int(seconds * info.samplerate), dtype="float32", always_2d=True).mean(axis=1)
    if (info.samplerate != rate or speed != 1.0) and len(audio):
        from fractions import Fraction

        from scipy.signal import resample_poly

        # Filtered resampling, so treble and hiss don't fold back into the clip.
        ratio = Fraction(rate / (info.samplerate * speed)).limit_denominator(1000)
        audio = resample_poly(audio, ratio.numerator, ratio.denominator).astype("float32")
    buf = io.BytesIO()
    sf.write(buf, audio, rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def audd_lookup(wav: bytes, token: str, retries: int = 2) -> dict | None:
    """Send one clip to AudD. Returns its 'result' (None when nothing matched)."""
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"api_token\"\r\n\r\n{token}\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"clip.wav\"\r\n"
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode() + wav + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(API_URL, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read())
            break
        except OSError:
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)
    if os.environ.get("TAPE_DEBUG"):
        print(f"\n  AudD replied: {json.dumps(data)[:500]}")
    if data.get("status") != "success":
        err = data.get("error", {})
        raise RuntimeError(f"AudD error {err.get('error_code')}: {err.get('error_message')}")
    return data.get("result")


def identify(master: Path, times: list[float], token: str, lookup=audd_lookup) -> list[Match]:
    out = []
    for t in times:
        res = lookup(clip_wav(master, t), token)
        if res:
            out.append(Match(t, res.get("artist", ""), res.get("title", ""), res.get("album", "")))
        else:
            out.append(Match(t))
    return out


def name_tracks(master: Path, tracks: list[Track], token: str, lookup=audd_lookup) -> tuple[list[Track], list[Match]]:
    """One clip a third of the way into each track; tracks the same song are merged."""
    times = [t.start + min(max(t.duration / 3, 5), max(t.duration - 15, 0)) for t in tracks]
    matches = identify(master, times, token, lookup)
    named = []
    for t, m in zip(tracks, matches):
        if named and m.key and m.key == named[-1][1]:
            # A pause inside one song was mistaken for a gap: join them back up.
            named[-1] = (Track(named[-1][0].start, t.end, named[-1][0].title), m.key)
        else:
            named.append((Track(t.start, t.end, m.label), m.key))
    return [t for t, _ in named], matches


def runs_of_songs(matches: list[Match]) -> list[list]:
    """Consecutive clips of the same song, as [first_time, last_time, key, label].

    Unrecognized clips are skipped, so they join whichever run they sit in.
    """
    runs: list[list] = []
    for m in matches:
        if m.key is None:
            continue
        if runs and runs[-1][2] == m.key:
            runs[-1][1] = m.at
        else:
            runs.append([m.at, m.at, m.key, m.label])
    return runs


def cuts_between(runs: list[list], change_points: list[float]) -> list[float]:
    """Where to cut between songs: the change point nearest halfway, else halfway."""
    cuts = []
    for prev, nxt in zip(runs, runs[1:]):
        lo, hi = prev[1], nxt[0]
        mid = (lo + hi) / 2
        inside = [c for c in change_points if lo < c < hi]
        cuts.append(min(inside, key=lambda c: abs(c - mid)) if inside else mid)
    return cuts


def refine(master: Path, result, token: str, every: float = 45.0, lookup=audd_lookup) -> tuple[list[Track], list[Match]]:
    """Use recognition to fix and name the tracks found by the splitter.

    Stretches with no gaps are sampled every `every` seconds and re-cut where
    the song changes; then every track still without a name gets one clip.
    """
    tracks = list(result.tracks)
    all_matches: list[Match] = []
    for a, b in result.gapless:
        times = list(np.arange(a + 10, max(b - 12, a + 10), every))
        matches = identify(master, times, token, lookup)
        all_matches += matches
        runs = runs_of_songs(matches)
        if len(runs) < 1:
            continue  # nothing recognized: keep the change-detection cuts
        inner = cuts_between(runs, [c for c in result.change_points if a < c < b])
        outside = [t for t in tracks if t.end <= a + 1 or t.start >= b - 1]
        edges = [a] + inner + [b]
        new = [Track(x, y, r[3]) for x, y, r in zip(edges, edges[1:], runs)]
        tracks = sorted(outside + new, key=lambda t: t.start)

    todo = [t for t in tracks if not t.title]
    if todo:
        _, matches = name_tracks(master, todo, token, lookup)
        all_matches += matches
        for t, m in zip(todo, matches):
            t.title = m.label

    # Neighbouring tracks that turned out to be the same song are joined.
    merged: list[Track] = []
    for t in tracks:
        if merged and t.title and t.title == merged[-1].title:
            merged[-1] = Track(merged[-1].start, t.end, t.title)
        else:
            merged.append(t)
    return merged, all_matches
