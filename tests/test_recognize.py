"""Recognition logic, with a fake AudD that knows where each song is."""

import numpy as np
import soundfile as sf

from tapedigitizer import cli, recognize
from tapedigitizer.splitter import SplitResult, Track

SR = 8000


def make_lookup(master, layout):
    """Map each clip back to its position by matching the clip's tone frequency."""
    calls = []

    def lookup(wav, token):
        import io

        audio, rate = sf.read(io.BytesIO(wav))
        spec = np.abs(np.fft.rfft(audio))
        hz = np.argmax(spec) * rate / len(audio)
        calls.append(hz)
        for _, _, f, artist, title in layout:
            if abs(hz - f) < 5:
                return {"artist": artist, "title": title} if title else None
        return None

    return lookup, calls


def tone_side(tmp_path, layout, total):
    t = np.arange(int(total * SR)) / SR
    audio = np.zeros(len(t), dtype="float32")
    for a, b, f, _, _ in layout:
        m = (t >= a) & (t < b)
        audio[m] = 0.2 * np.sin(2 * np.pi * f * t[m])
    p = tmp_path / "side.flac"
    sf.write(p, audio, SR)
    return p


def test_mix_region_recut_where_song_changes(tmp_path):
    layout = [(0, 200, 300, "A", "One"), (200, 410, 500, "B", "Two"), (410, 600, 700, "C", "Three")]
    p = tone_side(tmp_path, layout, 600)
    # Change detection found 260 (wrong), 197 and 400 (right).
    res = SplitResult([Track(0, 230), Track(230, 400), Track(400, 600)], "changes", False, [], [(0, 600)], [260, 197, 400])
    lookup, calls = make_lookup(p, layout)
    tracks, matches = recognize.refine(p, res, "tok", lookup=lookup)
    assert [t.title for t in tracks] == ["A - One", "B - Two", "C - Three"]
    # Cuts snap to the change point nearest halfway between differing clips.
    assert tracks[0].end == 197 and tracks[1].end == 400
    assert len(calls) == len(matches) == 13  # one clip every 45 s, from 10 s to 550 s


def test_tracks_named_and_false_gap_rejoined(tmp_path):
    layout = [(0, 100, 300, "A", "One"), (100, 220, 300, "A", "One"), (220, 300, 600, "", "")]
    p = tone_side(tmp_path, layout, 300)
    res = SplitResult([Track(0, 100), Track(100, 220), Track(220, 300)], "silence", True)
    lookup, calls = make_lookup(p, layout)
    tracks, _ = recognize.refine(p, res, "tok", lookup=lookup)
    # A pause inside "One" had split it; recognition joins it back. Unknown stays unnamed.
    assert [(t.start, t.end, t.title) for t in tracks] == [(0, 220, "A - One"), (220, 300, "")]
    assert len(calls) == 3


def test_import_two_mono_files(tmp_path):
    rng = np.random.default_rng(0)
    t = np.arange(SR * 100) / SR
    left = (0.2 * np.sin(2 * np.pi * 220 * t) + rng.normal(0, 1e-3, len(t))).astype("float32")
    sf.write(tmp_path / "3-Audio.aif", left, SR, subtype="PCM_24", format="AIFF")
    sf.write(tmp_path / "4-Audio.aif", left * 0.9, SR, subtype="PCM_24", format="AIFF")
    out = tmp_path / "work"
    assert cli.main(["import", "1", "a", str(tmp_path / "3-Audio.aif"), str(tmp_path / "4-Audio.aif"), "--out", str(out)]) == 0
    master = out / "Tape 001 - Side A" / "Tape 001 - Side A.flac"
    info = sf.info(master)
    assert info.channels == 2 and abs(info.duration - 100) < 0.01
    assert (out / "Tape 001 - Side A" / "tracks" / "01.flac").exists()
