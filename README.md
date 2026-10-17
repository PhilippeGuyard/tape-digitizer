# Tape digitizer

Records a cassette side hands-free from the Focusrite: it starts when the music
starts, stops after 20 seconds of silence (end of side), and saves a trimmed
24-bit FLAC master.

## The app

Double-click **Tape Digitizer.command** (in Finder, in this folder). The first
time, macOS may ask whether to open it: right-click it, choose Open, then Open.
A window opens with:

- the input in use and a level meter per input (click the input to change it),
- tape and side (filled in with the next number; a name like "Summer mix 1987"
  works too), and a Start button,
- live progress while a side records, splits and saves to Drive,
- a library of every side. Sides marked "Check splits" have guessed cuts: click
  one to see its waveform. Zoom in (− / +, pinch, or Ctrl + scroll) and scroll
  along; add a split at the playhead (S, or double-click), select a split to
  fine-tune or remove it (Delete), listen across it, and name the tracks.
  Undo / Redo (⌘Z / ⇧⌘Z) step through changes and "Reset all" goes back to the
  last save. "Save and upload" re-cuts the side and replaces it in Drive.
- Recordings are trimmed to the music: the leader before the first note and the
  run-out after the last are marked "Skipped", as are silences of 8 s or more.
  Any section can be skipped (talk, a mistake) with Skip on its row, or brought
  back with Keep. Skipped sections aren't saved and don't get a track number.
- "Save for later" keeps your edits without cutting or uploading anything
  (it also saves by itself a few seconds after each change). The side shows
  "Draft" in the library until you press "Save and upload".

So you can record side after side, and come back to the splits whenever you like.
Closing the Terminal window that opens with it quits the app.

## Setup (once)

    cd ~/code/tape-digitizer
    bash setup.sh

## Use

    ./tape devices          # list audio devices and how many inputs each has
    ./tape choose           # pick the device and inputs (e.g. 1,2 stereo or 1 mono); remembered
    ./tape levels           # live meter: set the gain so loud parts peak around -6 dB
    ./tape calibrate        # play blank tape for 10 s, get suggested thresholds
    ./tape record 42 A      # press play on the deck and walk away
    ./tape split 42 A       # re-split a side that's already recorded
    ./tape export 42 A      # re-save a side's MP3s to Drive

Each side then goes through three stages by itself: record, split, save to Drive.

- Work files stay on the Mac in `~/TapeDigitizer/work/Tape 042 - Side A/`: the
  full-side FLAC master, `tracks.txt` (where the splits are) and `tracks/` (FLAC).
- 320 kbps MP3s go to Google Drive in `Music/Tapes/Tape 042/Side A/`, tagged with
  album "Tape 042 Side A" and track numbers, and Google Drive uploads them.
- The full-side FLAC master is copied to `Music/Tapes/Masters/` (about 300 MB a
  side). To stop that, set `"masters_to_drive": false` in `~/TapeDigitizer/config.json`.
- `Music/Tapes/log.csv` lists every finished side, and flags the ones whose splits
  need checking.

Re-running `split` or `export` replaces that side's MP3s. Use `--no-upload` to
skip Drive, and `./tape set-drive <folder>` to save somewhere else.

## Bring in recordings you already made

Ableton records the left and right inputs as two mono files. Give both, left first:

    ./tape import 1 A "path/to/3-Audio ....aif" "path/to/4-Audio ....aif"

It becomes Tape 001 - Side A and is split like a fresh recording.

## How splitting works

After recording, the side is split automatically:

1. **Gaps**: normal tapes are cut in the middle of each silent gap between songs.
2. **Changes**: if a long side has no gaps (a seamless mix), it looks for points
   where the music changes character instead. These are estimates.

3. **Song recognition** (off by default; add `--recognize` to `split` or `import`,
   with your AudD API token in `audd_token.txt`). AudD found nothing on tape 1 A,
   so it's only worth trying on tapes of better-known releases. Stretches
   with no gaps are sampled every 45 s and cut where the song changes, and every
   track gets one 12-second clip looked up so it's named "Artist - Title".

If a split is uncertain, the side folder gets a `CHECK ME.txt`. To fix it, open
the master in Audacity, File > Import > Labels > `tracks.txt`, drag the labels
(you can also type song names into them), File > Export > Export Labels back over
`tracks.txt`, then:

    ./tape split 42 A --edited

To use a different source just once, add `--device` and `--inputs` to any
command, e.g. `./tape record 42 A --device Scarlett --inputs 3,4`.

Useful options for `record`: `--silence-db -48` if your tapes are noisy and a
side never ends, `--end-silence 30` if quiet passages end a side too early.

## Tests

    .venv/bin/pip install pytest && .venv/bin/python -m pytest
