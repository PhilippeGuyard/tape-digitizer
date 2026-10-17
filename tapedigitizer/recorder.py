"""Hands-free recording of one tape side.

The recorder waits for sound, records until the tape has been silent for a
while (end of side), then saves a trimmed FLAC master.

The decision logic lives in `SideDetector`, which only looks at blocks of
samples, so it can be tested without any audio hardware.
"""

from __future__ import annotations

import collections
import enum
import json
import queue
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf


def rms_db(block: np.ndarray) -> float:
    """RMS level of a block in dBFS (0 dB = full scale)."""
    if block.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))
    return 20 * np.log10(max(rms, 1e-6))


@dataclass
class RecorderSettings:
    samplerate: int = 48000
    block_seconds: float = 0.1
    # Sound above this level means the tape has started playing.
    start_threshold_db: float = -40.0
    # How long sound must last before we treat it as the tape starting.
    start_hold_seconds: float = 0.5
    # Audio kept from just before the start was detected.
    preroll_seconds: float = 2.0
    # Below this level counts as silence (tape hiss sits around -55 to -65 dB).
    silence_threshold_db: float = -50.0
    # This much continuous silence means the side has ended.
    end_silence_seconds: float = 20.0
    # Silence kept after the last sound when trimming the master.
    tail_seconds: float = 1.0
    # Safety stop, in case the end is never detected (C120 = 60 min a side).
    max_side_seconds: float = 65 * 60


class State(enum.Enum):
    WAITING = "waiting"
    RECORDING = "recording"
    DONE = "done"


class SideDetector:
    """Decides when a side starts and ends from a stream of audio blocks.

    `feed()` returns the blocks that belong in the recording (including
    pre-roll on the first one). After DONE, `last_sound_frame` says where
    the music actually ended so the silent tail can be trimmed.
    """

    def __init__(self, s: RecorderSettings):
        self.s = s
        self.state = State.WAITING
        self._preroll = collections.deque()
        self._preroll_frames = 0
        self._loud_run = 0.0
        self._silent_run = 0.0
        self.recorded_frames = 0
        self.last_sound_frame = 0
        self.level_db = -120.0
        self.end_reason = ""

    def feed(self, block: np.ndarray) -> list[np.ndarray]:
        if self.state is State.DONE:
            return []
        dur = len(block) / self.s.samplerate
        self.level_db = rms_db(block)

        if self.state is State.WAITING:
            self._preroll.append(block)
            self._preroll_frames += len(block)
            # Keep pre-roll before the sound began, not before it was confirmed.
            max_pre = int((self.s.preroll_seconds + self.s.start_hold_seconds) * self.s.samplerate)
            while self._preroll and self._preroll_frames - len(self._preroll[0]) >= max_pre:
                self._preroll_frames -= len(self._preroll.popleft())
            self._loud_run = self._loud_run + dur if self.level_db > self.s.start_threshold_db else 0.0
            if self._loud_run >= self.s.start_hold_seconds:
                self.state = State.RECORDING
                out = list(self._preroll)
                self._preroll.clear()
                for b in out:
                    self._account(b)
                return out
            return []

        # RECORDING
        self._account(block)
        if self._silent_run >= self.s.end_silence_seconds:
            self.state, self.end_reason = State.DONE, "silence"
        elif self.recorded_frames / self.s.samplerate >= self.s.max_side_seconds:
            self.state, self.end_reason = State.DONE, "max_length"
        return [block]

    def _account(self, block: np.ndarray) -> None:
        level = rms_db(block)
        self.recorded_frames += len(block)
        if level > self.s.silence_threshold_db:
            self._silent_run = 0.0
            self.last_sound_frame = self.recorded_frames
        else:
            self._silent_run += len(block) / self.s.samplerate

    @property
    def trimmed_frames(self) -> int:
        tail = int(self.s.tail_seconds * self.s.samplerate)
        return min(self.recorded_frames, self.last_sound_frame + tail)


@dataclass
class SideResult:
    master: str
    duration_seconds: float
    peak_db: float
    ended_by: str  # "silence", "max_length" or "stopped"
    recorded_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))


def record_side(
    out_dir: Path,
    name: str,
    settings: RecorderSettings,
    blocks,
    on_progress=None,
    should_stop=lambda: False,
) -> SideResult | None:
    """Record one side from an iterable of audio blocks.

    Writes a temporary WAV while recording, then saves `<name>.flac` trimmed
    to the end of the music, plus `<name>.json` with details. Returns None if
    no sound was ever detected.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f"{name}.recording.wav"
    det = SideDetector(settings)
    peak = 0.0
    ended_by = "stopped"
    writer = None
    try:
        for block in blocks:
            for b in det.feed(block):
                if writer is None:
                    writer = sf.SoundFile(tmp, "w", settings.samplerate, b.shape[1], "PCM_24")
                writer.write(b)
                peak = max(peak, float(np.max(np.abs(b))) if b.size else 0.0)
            if on_progress:
                on_progress(det)
            if det.state is State.DONE:
                ended_by = det.end_reason
                break
            if should_stop():
                break
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        return None

    master = out_dir / f"{name}.flac"
    frames = det.trimmed_frames
    with sf.SoundFile(tmp) as src, sf.SoundFile(master, "w", settings.samplerate, src.channels, "PCM_24", format="FLAC") as dst:
        remaining = frames
        while remaining > 0:
            chunk = src.read(min(remaining, settings.samplerate * 10))
            if len(chunk) == 0:
                break
            dst.write(chunk)
            remaining -= len(chunk)
    tmp.unlink()

    result = SideResult(
        master=master.name,
        duration_seconds=round(frames / settings.samplerate, 2),
        peak_db=round(20 * np.log10(max(peak, 1e-6)), 1),
        ended_by=ended_by,
    )
    (out_dir / f"{name}.json").write_text(json.dumps({"result": asdict(result), "settings": asdict(settings)}, indent=2))
    return result


def parse_inputs(text: str) -> tuple[int, ...]:
    """Turn "1,2", "3-4" or "1" into input numbers (1-based, as printed on the interface)."""
    nums: list[int] = []
    for part in text.replace(" ", "").split(","):
        if "-" in part:
            a, b = part.split("-")
            nums.extend(range(int(a), int(b) + 1))
        elif part:
            nums.append(int(part))
    if not nums or min(nums) < 1:
        raise ValueError(f"inputs must be numbers starting at 1, got {text!r}")
    if len(nums) > 2:
        raise ValueError("choose one input (mono) or two (stereo)")
    return tuple(nums)


def live_blocks(settings: RecorderSettings, device=None, inputs: tuple[int, ...] = (1, 2)):
    """Yield blocks from the chosen inputs of the audio interface, forever.

    `inputs` are 1-based input numbers; each yielded block has one column per input.
    """
    import sounddevice as sd

    if device is not None:
        available = sd.query_devices(device)["max_input_channels"]
        if max(inputs) > available:
            raise ValueError(f"{sd.query_devices(device)['name']} has {available} inputs, can't use input {max(inputs)}")
    cols = [i - 1 for i in inputs]
    q: queue.Queue = queue.Queue()

    def callback(indata, frames, t, status):
        if status:
            print(f"audio warning: {status}")
        q.put(indata[:, cols].copy())

    with sd.InputStream(
        samplerate=settings.samplerate,
        channels=max(inputs),
        device=device,
        dtype="float32",
        blocksize=int(settings.samplerate * settings.block_seconds),
        callback=callback,
    ):
        while True:
            yield q.get()


def list_inputs():
    """(index, name, input count, default rate) for every device that can record."""
    import sounddevice as sd

    return [
        (i, d["name"], d["max_input_channels"], int(d["default_samplerate"]))
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0
    ]


def find_device(hint: str = "focusrite|scarlett") -> int | None:
    """Index of the first input device whose name matches any of the hints."""
    words = [w.strip().lower() for w in hint.split("|")]
    for i, name, _, _ in list_inputs():
        if any(w in name.lower() for w in words):
            return i
    return None
