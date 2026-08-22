# Measuring the glasses

One command, one table. Everything else lands in files.

```sh
tools/glasses_measure.sh
```

Run it with the glasses plugged in **and on your head** — the panel powers
itself down without one, and from the phone that looks exactly like being
unplugged. The run refuses to start in that state rather than falling back to
the simulated panel, because a measurement taken against a stand-in looks
exactly like a measurement and is not one.

Some steps ask you to turn your head. The prompt appears in the table.

## Switching to the real hardware

Nothing to switch. The runner takes the real glasses unless told otherwise, and
checks for them the only way that cannot lie:

| check | why |
| --- | --- |
| the panel is in SurfaceFlinger's display list | a panel is scanning or it is not; there is no cache |
| the USB product id | informational only — it survives an unplug in `dumpsys usb`, which is how it was seen reporting a Pro 2 with nothing attached |

The app's simulated panel is only ever created by an explicit intent extra, and
the runner passes it only in `--mock`. There is no path where a real run reaches
one by accident.

## Before the first run

Push the test material and the benchmark once:

```sh
adb push ods_360_3d_tagged.mp4 vr180_3d_tagged.mp4 \
         motorist_original.mp4 ibiza_anaglyph.mp4 /sdcard/Movies/
adb push target/aarch64-linux-android/release/examples/library_bench /data/local/tmp/
```

`motorist_original.mp4` is *The Motorist revisited* by Ragnar di Marzo, CC BY-ND
4.0 from archive.org — real filmed stereoscopic 360 with its own `st3d` and
`sv3d` boxes. The other three are generated; see `tools/make_ods_reference.py`
and `tools/inject_spatial_metadata.py`.

Take the comparison baseline once, with the glasses **un**plugged:

```sh
tools/glasses_measure.sh --mock
```

Every real run afterwards prints a column showing where the two disagree.

## The steps

Run a subset by number: `tools/glasses_measure.sh 3 5`.

| # | measures | precondition | expected |
| --- | --- | --- | --- |
| 1 | product id, battery, saver | — | `0x1301` for a Pro 2. The battery state is recorded rather than judged: every timing below is read against it, and the same phone flat and charging is not the same computer. |
| 2 | panel modes in 2D **and** side-by-side, DisplayPort lanes | glasses worn | 2D offers 1920 wide at 60/30/20. Side-by-side offers 3840. **Anything above 60 Hz answers the open question** — the panel is rated for 120 in 2D and has never offered it, over four lanes with training successful. |
| 3 | pose rate, reader errors | glasses worn | 118.9 Hz, counted. Not 120: it is configured for 120 and delivers this, and the one per cent becomes a millisecond once a lookahead multiplies it. `errno 0`. |
| 4 | prediction lookahead, field of view | after step 3 | 9–11 ms on the real panel, from its own presentation deadline. 1920×1080 an eye at 25.8° vertical for a Pro 2. |
| 5 | frame gap idle / with a 4K sphere / while turning | **turn your head** when prompted | median under 16.7 ms, which is the panel. The interesting number is the worst case. |
| 6 | four formats, each read from the file alone | material pushed | `st3d=1` and full bounds for the 360s, `st3d=2` with `0.25` cropped each side for VR180. No layout or projection is passed — what is being measured is whether the file is understood from its own metadata. |
| 7 | parallax sign, anaglyph greyness, panel coverage | after step 6 | Parallax **negative**, within 3% of −149 px. The sign is the whole test: swapped eyes give the same magnitude with the opposite sign, everything looks identical, and nobody can wear it for ten minutes. |
| 8 | reading a media library | `library_bench` pushed | ~56 ns an entry. The parallel split is measured rather than assumed and has never paid on this hardware. |

## Reading the result

The table prints measured against expected. Below it, if a baseline exists, each
measurement appears beside the mock's.

Rows marked *differs — …* are supposed to: a virtual panel has its own refresh
rate, its own presentation deadline and no optics, so the timings and the
lookahead cannot match and the difference says something about the stand-in.

A row marked **DIFFERS, and should not** is the finding. It means the simulation
tells a story the hardware does not, and every hour spent developing against it
afterwards is spent on the wrong problem.

## Artefacts

`tools/measurements/<timestamp>/` holds `summary.txt`, `results.tsv`, the panel
captures as PNGs, and the raw logcat for each step. Nothing needs reading unless
a row disagrees.
