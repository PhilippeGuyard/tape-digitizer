"""Splitting simulated sides into tracks."""

import numpy as np
import pytest
import soundfile as sf

from tapedigitizer.splitter import (
    SplitSettings,
    Track,
    cut_tracks,
    find_tracks,
    read_labels,
    write_labels,
)

SR = 22050
rng = np.random.default_rng(0)


def hiss(seconds, db=-60):
    return rng.normal(0, 10 ** (db / 20), int(seconds * SR)).astype("float32")


def song(seconds, root_hz, bpm, bright=1.0, db=-18):
    """A crude 'song': a sustained chord plus a drum hit on every beat."""
    t = np.arange(int(seconds * SR)) / SR
    chord = sum(np.sin(2 * np.pi * root_hz * r * t) * bright ** k for k, r in enumerate([1, 1.25, 1.5, 2, 3]))
    beat = 60 / bpm
    env = np.exp(-((t % beat) / 0.05))
    drum = rng.normal(0, 1, len(t)) * env
    x = chord / 3 + 0.5 * drum
    x *= 10 ** (db / 20) / np.sqrt(np.mean(x**2))
    return (x + hiss(seconds)).astype("float32")


def write_side(tmp_path, parts):
    audio = np.concatenate(parts)
    p = tmp_path / "side.flac"
    sf.write(p, np.stack([audio, audio], axis=1), SR, subtype="PCM_24")
    return p, len(audio) / SR


def test_normal_tape_splits_on_gaps(tmp_path):
    lengths = [180, 210, 150, 240]
    parts = [hiss(2)]
    for i, n in enumerate(lengths):
        parts += [song(n, 110 * (1 + i / 4), 90 + 10 * i), hiss(3)]
    path, total = write_side(tmp_path, parts)

    res = find_tracks(path)
    assert res.method == "silence" and res.confident
    assert len(res.tracks) == 4
    # Cuts land in the middle of each 3 s gap.
    expected_cuts = np.cumsum([2 + lengths[0] + 1.5, 1.5 + lengths[1] + 1.5, 1.5 + lengths[2] + 1.5])
    got = [t.end for t in res.tracks[:-1]]
    assert np.allclose(got, expected_cuts, atol=0.3)


def test_short_quiet_passage_is_not_a_new_track(tmp_path):
    # A 2 s pause 20 s into a song: too short a 'track' to stand alone.
    parts = [hiss(2), song(20, 110, 100), hiss(2), song(200, 110, 100), hiss(3), song(180, 165, 120), hiss(5)]
    path, _ = write_side(tmp_path, parts)
    res = find_tracks(path)
    assert len(res.tracks) == 2


def test_seamless_mix_uses_change_detection(tmp_path):
    # Three songs with 4 s crossfades and no gaps.
    a, b, c = song(200, 110, 95, 0.4), song(180, 196, 128, 0.9), song(190, 147, 110, 0.6)
    xf = 4 * SR
    ramp = np.linspace(0, 1, xf, dtype="float32")

    def join(x, y):
        return np.concatenate([x[:-xf], x[-xf:] * ramp[::-1] + y[:xf] * ramp, y[xf:]])

    path, total = write_side(tmp_path, [join(join(a, b), c)])
    res = find_tracks(path)
    assert res.method == "changes" and not res.confident
    assert len(res.tracks) == 3
    cuts = [t.end for t in res.tracks[:-1]]
    assert abs(cuts[0] - 198) < 10 and abs(cuts[1] - 374) < 10


def test_labels_round_trip_and_point_labels(tmp_path):
    tracks = [Track(0, 100.5), Track(100.5, 250, "Blue Monday")]
    p = tmp_path / "tracks.txt"
    write_labels(p, tracks)
    back = read_labels(p)
    assert [(t.start, t.end) for t in back] == [(0, 100.5), (100.5, 250)]
    assert back[1].title == "Blue Monday"

    # Clicking once in Audacity makes point labels: start == end.
    p.write_text("60.0\t60.0\t\n130.0\t130.0\t\n")
    pts = read_labels(p, total=200)
    assert [(t.start, t.end) for t in pts] == [(0, 60), (60, 130), (130, 200)]


def test_cut_tracks(tmp_path):
    path, total = write_side(tmp_path, [song(30, 110, 100), song(20, 165, 120)])
    files = cut_tracks(path, [Track(0, 30), Track(30, total, "Second Song")], tmp_path / "tracks")
    assert [f.name for f in files] == ["01.flac", "02 - Second Song.flac"]
    assert abs(sf.info(files[1]).duration - 20) < 0.01


def test_split_command_then_edited_labels(tmp_path, capsys):
    from tapedigitizer import cli

    side = tmp_path / "Tape 003 - Side A"
    side.mkdir()
    audio = np.concatenate([hiss(2), song(100, 110, 100), hiss(3), song(90, 165, 120), hiss(5)])
    sf.write(side / f"{side.name}.flac", np.stack([audio, audio], 1), SR, subtype="PCM_24")

    assert cli.main(["split", "3", "a", "--out", str(tmp_path)]) == 0
    assert sorted(f.name for f in (side / "tracks").iterdir()) == ["01.flac", "02.flac"]

    # Edit in 'Audacity': add a name and a third cut.
    (side / "tracks.txt").write_text("0\t60\tIntro\n60\t103.5\tFirst\n103.5\t200\tSecond\n")
    assert cli.main(["split", "3", "a", "--out", str(tmp_path), "--edited"]) == 0
    names = sorted(f.name for f in (side / "tracks").iterdir())
    assert names == ["01 - Intro.flac", "02 - First.flac", "03 - Second.flac"]


def test_lead_in_run_out_and_long_silence_left_out(tmp_path):
    # 6 s leader, song, 15 s of silence mid-side, song, 20 s run-out.
    parts = [hiss(6, db=-70), song(100, 110, 100), hiss(15), song(90, 165, 120), hiss(20, db=-70)]
    path, total = write_side(tmp_path, parts)
    res = find_tracks(path)
    spans = [(round(t.start, 1), round(t.end, 1)) for t in res.tracks]
    assert len(spans) == 2
    assert abs(spans[0][0] - 5.5) < 0.3            # trimmed to 0.5 s before the music
    assert abs(spans[0][1] - (106 + 0.5)) < 0.5    # stops just into the long silence
    assert abs(spans[1][0] - (121 - 0.5)) < 0.5    # starts just before the next song
    assert abs(spans[1][1] - (211 + 1)) < 0.5      # 1 s kept after the music
