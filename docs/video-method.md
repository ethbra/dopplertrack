# The video's method, and what this fork adds

The method is colsto's (["I discovered a sound that leaks your exact location"](https://www.youtube.com/watch?v=UytYrhYLOKI));
this page maps each step of it to the code, describes what was added on top (the
step-by-step explainer, traced-line scoring, exports, countermeasures, the altitude datum,
Hermite paths, baro calibration, and a per-ray sound speed from the day's temperature
profile), and reports how those additions did on real data. The front page has the usage.

## The video's method, step by step, and where each step lives here

colsto released no code, and the video could not be fetched from this project's build
environment, so this list was assembled from the narration as quoted, with timestamps, by
independent re-implementations ([cbonitz/doppler-experiments](https://github.com/cbonitz/doppler-experiments),
which worked from the full transcript and three screenshots;
[franken_doppler_finder](https://github.com/Dicklesworthstone/franken_doppler_finder);
[gwendall/unheard](https://github.com/gwendall/unheard)).

| # | the video | here |
|---|---|---|
| 1 | Faint propeller tones (blade-pass, ~50-150 Hz) make a falling S-curve in a spectrogram, usually inaudible | `spectrogram` (whitened per bin), `detect` |
| 2 | An interactive panel builds the curve step by step: reception time 25.00 s, solve t_r = t_e + r(t_e)/c for t_e = 16.910 s, position, velocity, r = 2775 m, v_r = 121.16 m/s, f = 130 x 343/(343 + 121.16) = 96.07 Hz | `explain` prints the same steps for a real flight; `doppler.explain` reproduces the panel's numbers exactly (`tests/test_video_methods.py`) |
| 3 | Using the radial speed at reception time instead of emission time "makes an entirely wrong curve" | `curve_from_trajectory(..., model="reception")`, shown by `explain --plot`; never used for scoring |
| 4 | Read the curve by hand: the drop gives speed, its steepness the CPA distance, its midpoint the heard CPA time; then "generous ranges" | `fit` (automatic), `--v-tol`, `--d-tol` |
| 5 | Historical ADS-B (a paid Flightradar24 API), jets dropped | `fetch` (free adsb.lol), ICAO Doc 8643 engine classes |
| 6 | For every flight, a 200 m grid over a corridor around its track: CPA and the sound-delayed heard CPA time | `search` (flight-pair join, then lens lattices with exact CPA values) |
| 7 | Filters: the offset between the two events (920 s), speed and distance, then silence (no other plane heard in between); the video reports how many points survive each | the join and the lens gates; `search` logs the funnel (pairs from the join, then points and pairs after the exact gates and after silence) |
| 8 | Survivors grouped by flight pair; exact curves from the real, turning track; event 2 locked to event 1's clock; one pair scores far above the rest | locked scoring; `plot pairs` |
| 9 | Turns, speed and altitude changes are encoded in the curve and break ambiguities | real trajectories; `--interp hermite` makes turns between ADS-B reports curve (see below) |
| 10 | The 200 m grid is "a little too coarse": the right answer can be 141 m off and score badly | `refine` (50 m) |
| 11 | Street View at the best point shows the camera's surroundings | `export` (GeoJSON, KML, HTML map, each point with a Street View link) |
| 12 | A second example: one faint ~60 s flyover with a known time, scored by distance from a traced line | `search --utc`; `--score line` with `trace-line` (hand-traced CSV or the automatic ridge) |
| 13 | Countermeasures: a high-pass filter does not help (boosting brings it back); talking over it does not; it survives phone-call compression | `audit`, `scrub`; the claims are checked in `tests/test_countermeasures.py` |
| 14 | Still air, c = 343 m/s | `--temp`, `--wind`, `--weather` |

### Added in this round, and what is and is not validated

Every addition is opt-in; with no new flags the results are upstream's. The flight-data side
is checked on real ADS-B (`tests/test_real_adsb.py`, offline; `tests/test_real_day.py` for a
whole day) and the location side on both recordings (`tests/test_real_recordings.py`).

- **Altitude datum** (`--geoid`). ADS-B geometric altitude is GNSS height above the WGS-84
  ellipsoid; terrain tiles give height above mean sea level. The two differ by the geoid
  undulation (about -33 m around New York, -28 m in south-west Florida), so every observer
  sat that far too high relative to the aircraft. The sample trace in `tests/data` shows it
  independently: its GNSS altitude minus its QNH-corrected pressure altitude is -32.2 m
  (median of 1,025 reports) where EGM96 gives -33.6 m. Observers are now placed at terrain
  height plus the EGM96 undulation (NGA, public domain, 2.7 MB, downloaded once from PROJ's
  data repository), and baro-only aircraft get the same shift unless `--calibrate-baro`
  maps them onto the GNSS datum directly. At Oyster Bay this moves the observer 32 m down,
  which changes the King Air's 3.6 km slant range by about 9 m.

- **`explain`.** The video's step-by-step panel for any flight, observer and reception time,
  including what the reception-time model would say. On the video's own numbers it gives
  t_e = 16.910 s, r = 2775 m, v_r = 121.16 m/s and f = 96.07 Hz.
- **Filter funnel.** `search` logs, after the join, how many points and flight pairs pass the
  exact speed, distance and offset gates and then the silence filter: the counts the video
  quotes. (The video's first stage, the offset filter alone over a grid, has no equivalent
  since 0.3.0 starts from flight pairs rather than a grid.)
- **Recording clock.** Every search and refine row has `rec_start_unix`, the UTC time of
  the recording's first sample that the candidate implies.
- **Turn-aware trajectories** (`--interp hermite`). ADS-B reports arrive every 1-20 s; joined
  with straight lines, a turning aircraft's velocity is constant between reports and jumps at
  each one, so the modelled range rate is a staircase. Each readsb report also carries ground
  speed and true track; using them as the velocity at each report (cubic Hermite in lat/lon,
  monotone cubic in altitude) makes the path curve the way the aircraft did. On a 1.5 deg/s
  turn at 60 m/s, with exact speed and track in each report:

  | report interval | linear: position, curve error | hermite: position, curve error |
  |---|---|---|
  | 4 s | 3.1 m, 0.48 Hz | 0.06 m, 0.01 Hz |
  | 8 s | 12.6 m, 1.4 Hz | 0.12 m, 0.02 Hz |
  | 15 s | 44 m, 3.1 Hz | 0.27 m, 0.03 Hz |

  (100 Hz tone, observer 4 km off the turn.) A 1.4 Hz error is as wide as the ridge score's
  whole 1.5 % tolerance, so this is the change most likely to matter on turning passes such
  as the faint N7681Y curve above. Real reports carry quantised and noisy speed and track,
  so the gain on real data is smaller: on the sample trace, predicting every other report
  from the rest, the median horizontal error falls from 6.7 m to 4.7 m (in turns faster
  than 0.5 deg/s, 7.4 to 4.4 m).

  ("Hermite" is the name of the interpolation: a cubic between two points that matches both
  the positions *and* the velocities given at them. Linear interpolation matches only the
  positions, so the path has a corner, and the velocity a jump, at every report.)
- **Baro datum** (`--calibrate-baro`). Geometric altitude is GNSS height; barometric altitude
  assumes a standard atmosphere and is off by tens of metres on any non-standard day (76 m on
  the test trace, QNH 1026 hPa). Aircraft that report both measure the offset; it is fitted per
  half hour as a + b x baro and applied to aircraft that only report baro. On the sample
  trace, hiding one 10-minute block of GNSS altitude at a time: raw pressure altitude is 76 m
  off, calibrated 0.8 m (median; 12 m at the 90th percentile). That calibrates the aircraft
  from its own other blocks, which flatters it; `tests/test_real_day.py` does it across
  aircraft.
- **Weather** (`--temp`, `--lapse`, `--wind SPEED FROM_DEG`, `--weather`). The speed of sound
  follows temperature (0.6 m/s per degree C) and, since temperature falls with height, differs
  from ray to ray: each ray uses the mean over the heights it crosses (see Real-data results
  below for why). A uniform wind is modelled exactly: the sound heard at t_e + tau left when
  |D + w tau| = c tau (D from observer to aircraft), and the heard frequency is
  f0 / (1 + d tau / d t_e). The obvious shortcut (a direction-dependent sound speed c + w.k
  with the ray straight) is off by about as much as ignoring the wind, because the ray tilts
  by about |w|/c and that tilt multiplies the aircraft's speed; the test checks the model
  against an exact air-frame ray solution. `--weather` reads the hour's temperature profile
  and the mean wind below 3 km from Open-Meteo's historical forecast archive (pressure levels).
- **Traced-line scoring** (`--score line`, `trace-line`). The video scores its faint curve by
  distance from a line drawn on the spectrogram. `trace-line` stores a line in the event JSON,
  from a hand-traced CSV (any tool that exports t, f points) or from the automatic ridge's
  reliable frames, which a person can then trim; `--score line` scores against it with the
  ridge score's kernel.
- **Exports and figures.** `export` writes the best point of each flight pair as GeoJSON, KML
  or a standalone Leaflet map, each with Google Maps and Street View links, the video's
  final check. `plot` draws an event fit, the score map, the best score per flight pair, and
  a candidate's modelled curves over both event windows.
- **Peak spread.** `refine` prints how many fine points score within 1 % and 5 % of the peak
  and how far the farthest is: a direct read of how well the events pin the point (tight for
  a low lateral pass, wide for two overhead ones, as on the eagle cam).
- **Indexing saved traces** (`fetch --from-traces day1.npz day2.npz`), including several
  days for a recording whose UTC day is uncertain or that spans midnight UTC, and the
  `--interp` and `--calibrate-baro` choices, which are baked into the day index.

### Countermeasures

The video's last part, checked on a synthetic recording (`tests/test_countermeasures.py`):

- A 4th-order high-pass at 200 Hz, above both tones, followed by a 20 dB boost: the fit lands
  on the same curve. A linear filter scales tone and noise in a frequency bin alike, and the
  per-bin whitening undoes the scale.
- A 300-3400 Hz phone band removes a 120 Hz fundamental entirely; its fourth harmonic, at
  480 Hz, traces the same Doppler factor and fits to the right speed and CPA time.

`audit` reports whether a recording leaks: it runs `detect` and compares each flyover with
the best score the detector finds in the *time-reversed* spectrogram. A Doppler curve only
ever falls, so reversed it is a rising curve the model cannot fit, while noise, bursts, hum
and speech look the same either way; the reversed best is the level chance reaches in that
recording. Two flyovers above it is "high" (enough for a blind two-event search), one is
"medium" (enough with a known time). `detect --null` prints the same chance level.

`scrub` removes the leak from the audio of a given list of events (`audit --out` writes
one). In the default `fill` mode the short-time spectrum along every harmonic of each curve,
up to 1.5 kHz, is replaced by that bin's own noise floor with random phase. Zeroing the bins
instead would leave a dark curve of the same shape, which leaks just as much. `mute` silences
each event window and `cut` deletes it, which also destroys the time between events the
two-event search depends on. Only the events passed in are removed, and a detected fit
can be far enough off the real tone that the mask misses it (on the synthetic recording
`detect` fitted one flyover at 103 Hz and 44 m/s, where the tone was 120 Hz at 110 m/s), so
`scrub --band FLO FHI` audits its own output, which fits the remaining tones afresh, and
scrubs whatever is still above chance, up to `--passes` more times. On that recording
the risk went from high to medium after the first pass and to low after the second.

## Validation

| tests | data | runs |
|---|---|---|
| `test_physics`, `test_video_methods`, `test_countermeasures`, `test_pipeline_synthetic`, ... | synthetic, generated in the test | always |
| `test_real_adsb` | the real readsb trace and an EGM96 crop in `tests/data` | always (offline) |
| `test_real_day` | one real day of ADS-B over a region | `DOPPLERTRACK_DAY_TRACES=traces.npz` |
| `test_oyster_bay`, `test_real_recordings` | the two recordings with known answers, and their days of ADS-B | `DOPPLERTRACK_OYSTER_BAY=dir`, `DOPPLERTRACK_SWFL=dir` |

`test_real_recordings` searches around each known answer under six option sets (the old
baseline, the geoid, Hermite, Hermite with baro calibration, and the temperature profile with
and without wind, from the saved weather in `tests/data`) and asserts the right flight pair
stays first and within 500 m (Oyster Bay) or 2.5 km (the eagle cam, two overhead passes); with `DOPPLERTRACK_REPORT=report.csv` it writes
the peak distance, score and margin for each, which is the evidence for changing a default.
Preparing the data (needs ffmpeg and yt-dlp; each day of ADS-B streams ~4 GB):

```bash
mkdir -p data/oyster_bay data/swfl
yt-dlp -f bestaudio -o data/oyster_bay/rec.%(ext)s https://youtu.be/4CxVxuMVkMc
uv run dopplertrack spectrogram data/oyster_bay/rec.* data/oyster_bay/rec.npz
uv run dopplertrack fetch 2026-09-08 data/oyster_bay/day_ny.npz --bbox 40.4 41.3 -74.4 -72.6 --traces data/oyster_bay/traces_ny.npz
yt-dlp -f bestaudio -o data/swfl/rec.%(ext)s https://youtu.be/zs2o5Rhlp5w
uv run dopplertrack spectrogram data/swfl/rec.* data/swfl/rec.npz
uv run dopplertrack fetch 2024-12-28 data/swfl/day_swfl.npz --bbox 26.115 27.285 -82.505 -81.195 --traces data/swfl/traces_swfl.npz
DOPPLERTRACK_OYSTER_BAY=data/oyster_bay DOPPLERTRACK_SWFL=data/swfl \
  DOPPLERTRACK_DAY_TRACES=data/oyster_bay/traces_ny.npz DOPPLERTRACK_REPORT=report.csv \
  uv run pytest -s tests/test_real_recordings.py tests/test_real_day.py tests/test_oyster_bay.py
```

### Real-data results (2026-11-05)

Both recordings and their days of ADS-B were supplied by the repository owner (Oyster Bay:
1,016 non-jet flights; eagle cam: 556). Flight-data checks, the aircraft against their own
reports:

| check | Oyster Bay day | eagle-cam day |
|---|---|---|
| Hermite vs linear, every other report held out (median / 90th percentile) | 5.9 / 15.0 -> 4.2 / 11.1 m (730,342 reports) | 8.4 / 21.9 -> 5.9 / 16.3 m (183,883) |
| Baro calibration across aircraft, GNSS altitude hidden (median error) | 68.6 -> 11.4 m (174 aircraft) | 68.6 -> 2.7 m (96) |
| Airborne reports carrying GNSS altitude | 99 % | 99 % |

**The speed of sound is per ray.** Scored with one sound speed for both flyovers, the
Oyster Bay PC-12 (5.5 km above the observer) fits best at 337 m/s and the King Air (1.1 km)
at 342 m/s. That afternoon's temperature profile (Open-Meteo pressure levels: 26 C at the
ground, 13 C at 1.6 km) gives column means of 336 and 344 m/s. So `c` can now be a
`doppler.Atmosphere`: each ray travels at the mean sound speed of the air it crosses, and
`--weather` fetches the hour's profile (and the mean wind below 3 km); `--lapse` builds a
standard-lapse column from `--temp`.

Location, searching around each known answer with the committed event fits (Oyster Bay 10 m
grid, eagle cam 50 m; "score" is the true pair's best score):

| configuration | Oyster Bay: peak, score | eagle cam: peak, score |
|---|---|---|
| baseline: 15 C, old datum, linear | 66 m, 29.09 | 1,284 m, 10.21 |
| geoid + Hermite, 15 C | 80 m, 28.41 | 1,674 m, 10.02 |
| geoid + Hermite, the afternoon's surface temperature for every ray | 166 m, 24.39 | 1,303 m, 9.30 |
| geoid + Hermite + profile | 105 m, 28.70 | 1,613 m, 10.24 |
| geoid + Hermite + profile + wind | 89 m, 29.02 | 1,613 m, 10.17 |
| old datum + linear + profile | 118 m, 28.29 | 1,739 m, 10.27 |

What the two recordings say:

- **None of the options moves the answer by a meaningful amount.** Every configuration finds
  the right flight pair; Oyster Bay stays within 66-166 m, and the eagle cam stays at the
  1.3-1.7 km its two overhead passes allow (the README's earlier 592 m used three more
  flyovers).
- **The obvious use of weather is harmful.** The afternoon's surface temperature (26 C) as the
  sound speed of every ray cuts the Oyster Bay score by 16 % and moves its peak to 166 m.
  The old 15 C default works only because 340 m/s happens to sit between the two rays' true
  means. The profile recovers the fit (28.70-29.02) for physical reasons rather than by luck.
- **Errors were cancelling.** With a constant 15 C, the geoid fix scores slightly worse on both
  recordings; with the profile, it scores better at Oyster Bay (28.70 vs 28.29, peak 105 vs
  118 m). The old datum was partly compensating for the wrong sound speed.

Defaults are therefore unchanged except the geoid: `--interp hermite`, `--calibrate-baro`,
`--weather` and `--lapse` are measurably right about the flight data and the atmosphere but
have not improved a location on these two recordings, so they stay opt-in. Two recordings
with known answers are too few to settle more than that; lateral, low passes (which constrain
the location) are what the next validation recordings need.

### Livermore: 40 single flyovers (`benchmarks/livermore.py`)

A public dataset (Pathak 2026, [Zenodo 23424278](https://zenodo.org/records/23424278),
CC BY 4.0): 40 smartphone recordings of aircraft over Livermore, CA, May 2026, each with the
author's traced frequency curve (UTC timestamps), the ADS-B track, and the true horizontal,
vertical and slant distance from the recorder to the track. The recorder's position is not
published and it moved between sessions, so each configuration locates the recorder from one
traced curve (free clock shift and rest frequency) and is scored by the slant distance it
recovers. The measurement is the author's, so this tests the propagation model alone.

| configuration | median slant error | mean | better than old (of 40) | paired p |
|---|---|---|---|---|
| old: 15 C, linear, old datum | 84 m | 136 m | n/a | n/a |
| geoid | 83 m | 132 m | 18 | 0.61 |
| geoid + Hermite | 81 m | 136 m | 21 | 0.25 |
| + the afternoon's surface temperature | 69 m | 122 m | 24 | 0.44 |
| + temperature profile | 56 m | 134 m | 22 | 0.37 |
| + profile + wind | 63 m | 138 m | 22 | 0.43 |

The profile lowers the median by a third, but no configuration is distinguishable from the
old model (Wilcoxon signed-rank, every p > 0.2). The traced curves sit about 3.3 Hz (1 %) from
every model's best fit, far above the fractions of a hertz the changes make, so on this data
the measurement noise decides; the aircraft here are also low (median 1.3 km), where the
profile differs little from the surface. Real-data evidence so far, over 42 flyovers on three
sites: the changes fix the physics they target and do not yet show in a location.

