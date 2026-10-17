"""Listening to the input through the Mac's sound output."""
import numpy as np

from tapedigitizer.app import Monitor


def play(m, frames):
    out = np.ones((frames, 2), dtype="float32")
    m._play(out, frames, None, None)
    return out


def test_mono_input_plays_on_both_sides_at_the_set_volume():
    m = Monitor(48000)
    m.on, m.volume = True, 0.5
    m.feed(np.full((100, 1), 0.4, dtype="float32"))
    out = play(m, 150)
    assert np.allclose(out[:100], 0.2)
    assert np.all(out[100:] == 0)  # runs dry: silence, not old audio


def test_nothing_is_buffered_while_off_and_the_buffer_never_lags():
    m = Monitor(1000)
    m.feed(np.ones((100, 2), dtype="float32"))
    assert len(m.buf) == 0
    m.on = True
    for _ in range(20):
        m.feed(np.ones((100, 2), dtype="float32"))
    assert len(m.buf) == int(Monitor.MAX_BUFFER * 1000)
