"""Simulated tape sides, so the recorder can be checked without hardware."""

import json

import numpy as np
import soundfile as sf

from tapedigitizer.recorder import RecorderSettings, record_side

SR = 8000  # low rate keeps the tests fast


def hiss(seconds, db=-60, rng=np.random.default_rng(0)):
    n = int(seconds * SR)
    return rng.normal(0, 10 ** (db / 20), (n, 2)).astype("float32")


def music(seconds, db=-15):
    t = np.arange(int(seconds * SR)) / SR
    tone = (10 ** (db / 20) * np.sqrt(2) * np.sin(2 * np.pi * 440 * t)).astype("float32")
    return np.stack([tone, tone], axis=1) + hiss(seconds)


def blocks(audio, block=800):
    for i in range(0, len(audio), block):
        yield audio[i : i + block]


def settings(**kw):
    return RecorderSettings(samplerate=SR, end_silence_seconds=5, **kw)


def test_records_side_and_stops_on_silence(tmp_path):
    # 3 s before play, two songs with a 2 s gap, then the side runs out.
    tape = np.concatenate([hiss(3), music(10), hiss(2), music(8), hiss(30)])
    res = record_side(tmp_path, "side", settings(), blocks(tape))

    assert res.ended_by == "silence"
    data, sr = sf.read(tmp_path / "side.flac")
    # pre-roll (2 s) + 10 + 2 + 8 + tail (1 s) = 23 s; the song gap is kept.
    assert abs(len(data) / sr - 23) < 0.3
    assert not (tmp_path / "side.recording.wav").exists()
    meta = json.loads((tmp_path / "side.json").read_text())
    assert meta["result"]["ended_by"] == "silence"


def test_brief_click_does_not_start_recording(tmp_path):
    tape = np.concatenate([hiss(2), music(0.2), hiss(10)])
    assert record_side(tmp_path, "side", settings(), blocks(tape)) is None


def test_long_side_hits_safety_limit(tmp_path):
    tape = music(20)
    res = record_side(tmp_path, "side", settings(max_side_seconds=10), blocks(tape))
    assert res.ended_by == "max_length"
    assert abs(res.duration_seconds - 10) < 0.3


def test_quiet_passage_shorter_than_limit_keeps_recording(tmp_path):
    # A 4 s quiet passage mid-song must not end the side (limit is 5 s).
    tape = np.concatenate([music(5), hiss(4), music(5), hiss(10)])
    res = record_side(tmp_path, "side", settings(preroll_seconds=0), blocks(tape))
    assert abs(res.duration_seconds - 15) < 0.3
