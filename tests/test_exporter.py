"""Saving tracks to the (pretend) Google Drive folder."""

import csv

import numpy as np
import soundfile as sf

from tapedigitizer import cli

SR = 22050


def make_side(root, name="Tape 001 - Side A"):
    side = root / name / "tracks"
    side.mkdir(parents=True)
    (root / name / f"{name}.flac").write_bytes(b"master")
    t = np.arange(SR * 5) / SR
    for i, stem in enumerate(["01", "02 - The Message"]):
        x = 0.2 * np.sin(2 * np.pi * (220 + 110 * i) * t)
        sf.write(side / f"{stem}.flac", np.stack([x, x], 1), SR, subtype="PCM_24")
    return side.parent


def test_export_writes_tagged_mp3s_and_log(tmp_path, isolated):
    work = tmp_path / "work"
    make_side(work)
    assert cli.main(["export", "1", "A", "--out", str(work)]) == 0

    folder = isolated / "Tape 001" / "Side A"
    mp3s = sorted(p.name for p in folder.glob("*.mp3"))
    assert mp3s == ["01.mp3", "02 - The Message.mp3"]
    f = sf.SoundFile(folder / "02 - The Message.mp3")
    assert (f.title, f.album, f.tracknumber) == ("The Message", "Tape 001 Side A", "2")
    assert abs(f.frames / f.samplerate - 5) < 0.1

    rows = list(csv.DictReader((isolated / "log.csv").open()))
    assert [(r["tape"], r["side"], r["tracks"]) for r in rows] == [("Tape 001", "A", "2")]
    assert (isolated / "Masters" / "Tape 001 - Side A.flac").read_bytes() == b"master"


def test_reexport_replaces_tracks_and_log_row(tmp_path, isolated):
    work = tmp_path / "work"
    side = make_side(work)
    cli.main(["export", "1", "A", "--out", str(work)])
    (side / "tracks" / "02 - The Message.flac").unlink()
    cli.main(["export", "1", "A", "--out", str(work)])
    make_side(work, "Tape 002 - Side B")
    cli.main(["export", "2", "b", "--out", str(work)])

    assert [p.name for p in (isolated / "Tape 001" / "Side A").glob("*.mp3")] == ["01.mp3"]
    rows = list(csv.DictReader((isolated / "log.csv").open()))
    assert [(r["tape"], r["side"], r["tracks"]) for r in rows] == [("Tape 001", "A", "1"), ("Tape 002", "B", "2")]


def test_renaming_a_side_moves_it_in_the_work_folder_and_in_drive(tmp_path, isolated):
    import time
    from tapedigitizer import app

    work = tmp_path / "work"
    make_side(work)
    assert cli.main(["export", "1", "A", "--out", str(work)]) == 0

    eng = app.Engine.__new__(app.Engine)  # no audio device needed for this
    eng.work_dir, eng.lock, eng.status = work, app.threading.Lock(), "idle"
    assert eng.rename_side("Tape 001 - Side A", "Summer mix 1987", "b") == "Summer mix 1987 - Side B"
    for _ in range(100):
        if eng.status != "saving":
            break
        time.sleep(0.1)
    assert eng.status == "done", eng.message

    new = work / "Summer mix 1987 - Side B"
    assert (new / "Summer mix 1987 - Side B.flac").exists() and not (work / "Tape 001 - Side A").exists()
    assert not (isolated / "Tape 001").exists()
    assert sorted(p.name for p in (isolated / "Summer mix 1987" / "Side B").glob("*.mp3")) == ["01.mp3", "02 - The Message.mp3"]
    assert (isolated / "Masters" / "Summer mix 1987 - Side B.flac").exists()
    assert not (isolated / "Masters" / "Tape 001 - Side A.flac").exists()
    with (isolated / "log.csv").open(newline="") as f:
        assert [(r["tape"], r["side"]) for r in csv.DictReader(f)] == [("Summer mix 1987", "B")]


def test_tapes_with_no_name_get_their_own_running_number(tmp_path):
    from tapedigitizer import app

    eng = app.Engine.__new__(app.Engine)
    eng.work_dir = tmp_path
    assert eng.unnamed_side("A") == "Unnamed 001 - Side A"
    (tmp_path / "Unnamed 001 - Side A").mkdir()
    (tmp_path / "Tape 007 - Side A").mkdir()
    assert eng.unnamed_side("B") == "Unnamed 001 - Side B"  # the other side of the same tape
    assert eng.unnamed_side("A") == "Unnamed 002 - Side A"  # a new tape
