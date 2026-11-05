# Changelog

## Unreleased

- Traces keep what each report says (roadmap D4): the altitude's source (geometric,
  barometric, none, honouring readsb's flag 8 for a geometric altitude in slot 3), the
  pressure altitude, the position source (ADS-B, ADS-R, TIS-B, MLAT, ADS-C), the row flags,
  track and vertical rate. `alt` is unchanged. Trace files gain six rows; older files still
  load. A test shows on the sample trace why the source matters: geometric altitude minus
  QNH-corrected pressure altitude is -32 m, the EGM96 geoid there (-34 m), not zero.

- `dopplertrack.verdict`: the winner of a search is judged against the wrong
  pairs the same search scored (an exponential upper tail fitted on the best
  wrong pairs; p is the chance a wrong winner would lead the next wrong pair by
  the observed gap). `search` prints it after the top table: located, ambiguous,
  no match, or insufficient. On the saved results two events alone are always
  ambiguous, the extra flyovers make both recordings located, and the box moved
  150 km from the answer is no match. Those tables are now under docs/results
  and tests/test_verdict.py checks them.

- `detect` flags events that coincide with a broadband burst (`burst`: the
  fraction of the 3 min around the CPA in which the whole band is loud at
  once, three robust sigmas above its median level). On a railcam recording
  every top detection was a locomotive passing the camera, whose engine tone
  sweeps like a low-altitude flyover and arrives with the rumble.
  `--max-burst` drops them.

- A third ground truth, a 12-hour Railstream camera in Atlanta, searched
  blind: negative. The true pairs rank 6–32 nationwide within 440 m and never
  win; the archived stream has a 107 s dropout that puts events on two clocks;
  non-aircraft extras bury the right pair; and two of twenty verdicts with no
  right answer were "located" (docs/experiments.md). Its tables are in
  docs/results and the two false positives are expected failures in
  tests/test_verdict.py.

## 0.3.0 (2026-11-04)

The search no longer needs a box. One day of flights over a continent is
indexed once; flight pairs are joined on the geometry of the two events, ground
points are laid only in each pair's lens, and the same exact CPA gates, silence
filter and locked scorer rank them. Validated on both ground truths over the
whole continental US: Oyster Bay ranks the true pair 1 of 9,669 (32 m); the
eagle cam needs its further detected flyovers (`--extra`) and then ranks 1 of
13,298 (627 m). See docs/experiments.md.

**Breaking: the CLI is six commands.**

| before | now |
|---|---|
| `fetch-day DATE traces.npz --bbox …`, `extract-region DIR traces.npz --bbox …`, `index-day traces.npz day.npz` | `fetch DATE day.npz [--bbox …] [--archive DIR] [--traces traces.npz]`, which writes the day index directly |
| `search … --traces traces.npz --bbox …` (200 m grid over a box) | `search … --day day.npz [--bbox …]` (the join; `--bbox` restricts it) |
| `search … --point LAT LON` | `search … --day day.npz --point LAT LON` |
| `search-wide … --index day.npz` | `search … --day day.npz` |
| `search-single … --event EV --utc T` | `search … --events EV --utc T` |
| `refine … --traces traces.npz --bbox …` | `refine … --day day.npz` |

**Breaking: the library.** `dopplertrack.wide` is merged into
`dopplertrack.search` (`DayIndex`, `pair_join`, `single_join`, `lens_candidates`,
`score_candidates`, `run_search`, `rescore_with_extra`, `pick_flight`). The grid
search (`cpa_table`, `pair_candidates`, the CPA cache, `Grid.points`) and
`pipeline.run_search`, `run_single_search`, `extract_region`, `fetch_region`,
`segments_for_search`, `relocate_row` are gone; `pipeline.fetch` builds a day
index, `pipeline.refine` takes one. `build_segments` lost `clip_enu`. Result
CSVs carry `seg1`/`seg2` into the index they were made from, and `refine` checks
them against it.

Also:

- The locked scorer samples the spectrogram window along time once per event
  and walks the sweep in small chunks: twice as fast on one core, eleven times
  on sixteen (it was memory-bandwidth-bound in a process pool).
- The lens stage computes exact CPA with a KD-tree nearest-sample query instead
  of a dense distance matrix (12 min → 71 s for a CONUS day).
- `load_traces` reads the metadata member once (371 s → 5 s on 35k traces).
- Terrain tiles are written atomically and re-fetched if corrupt, so a process
  pool can share the cache.
- One BLAS thread per process is set before numpy loads in the CLI.

## 0.2.0 (2026-11-03)

- Coverage-weighted scoring: brightness summed over the whole measured window,
  so a flight that leaves the data mid-pass is not judged on its bright half.
- `refine --extra`: further detected flyovers scored on the same clock.
- `detect`: automatic Doppler event detection over a whole recording.
- `fetch-day`: stream an adsb.lol day archive from GitHub without storing it.
- Eagle-cam validation (second ground truth).

## 0.1.0 (2026-11-03)

- Spectrogram, curve fit, regional grid search, refine; Oyster Bay result.
