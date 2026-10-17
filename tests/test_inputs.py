"""Device and input selection, with a fake audio interface standing in for the Focusrite."""

import sys
import types

import numpy as np
import pytest
import soundfile as sf

from tapedigitizer import cli
from tapedigitizer.recorder import parse_inputs

SR = 48000


class FakeStream:
    """Plays a 4-input recording: tape on input 3 only, hiss elsewhere."""

    def __init__(self, samplerate, channels, device, dtype, blocksize, callback):
        assert channels == 3  # opened just wide enough for input 3
        self.cb, self.n = callback, blocksize
        rng = np.random.default_rng(1)
        t = np.arange(int(SR * 6)) / SR
        music = (0.2 * np.sin(2 * np.pi * 440 * t)).astype("float32")
        sig = np.concatenate([np.zeros(SR), music, np.zeros(SR * 25)]).astype("float32")
        self.audio = rng.normal(0, 1e-4, (len(sig), 4)).astype("float32")
        self.audio[:, 2] += sig

    def __enter__(self):
        for i in range(0, len(self.audio), self.n):
            self.cb(self.audio[i : i + self.n, :3], self.n, None, None)
        return self

    def __exit__(self, *a):
        pass


@pytest.fixture
def fake_sd(monkeypatch, tmp_path):
    devices = [
        {"name": "MacBook Air Microphone", "max_input_channels": 1, "default_samplerate": 48000},
        {"name": "Scarlett 4i4 USB", "max_input_channels": 4, "default_samplerate": 48000},
    ]
    mod = types.SimpleNamespace(
        query_devices=lambda i=None: devices if i is None else devices[i],
        InputStream=FakeStream,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", mod)
    monkeypatch.setattr(cli, "CONFIG_FILE", tmp_path / "config.json")
    return tmp_path


def test_parse_inputs():
    assert parse_inputs("1,2") == (1, 2)
    assert parse_inputs("3-4") == (3, 4)
    assert parse_inputs("2") == (2,)
    with pytest.raises(ValueError):
        parse_inputs("0")
    with pytest.raises(ValueError):
        parse_inputs("1-3")


def test_record_from_chosen_mono_input(fake_sd, monkeypatch):
    # "choose" saves the Scarlett, input 3; "record" then uses it without flags.
    answers = iter(["1", "3"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    cli.main(["choose"])

    out = fake_sd / "work"
    assert cli.main(["record", "7", "b", "--out", str(out), "--end-silence", "5"]) == 0
    data, sr = sf.read(out / "Tape 007 - Side B" / "Tape 007 - Side B.flac")
    assert data.ndim == 1  # mono, from input 3 only
    assert abs(len(data) / sr - 8) < 0.3  # 1 s lead-in + 6 s music + 1 s tail


def test_too_many_inputs_rejected(fake_sd):
    with pytest.raises(ValueError, match="has 1 inputs"):
        cli.main(["levels", "--device", "0", "--inputs", "1,2"])


def test_saved_device_found_by_name_after_renumbering(fake_sd, monkeypatch):
    answers = iter(["1", "3"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    cli.main(["choose"])
    # Plug in another device so the Scarlett moves from number 1 to number 2.
    devs = sys.modules["sounddevice"].query_devices()
    devs.insert(0, {"name": "AirPods", "max_input_channels": 1, "default_samplerate": 48000})
    args = cli.argparse.Namespace(device=None, inputs=None)
    assert cli.resolve_source(args) == (2, (3,))
