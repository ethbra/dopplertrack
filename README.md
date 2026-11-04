# dopplertrack

Acoustic geolocation from the Doppler curves of propeller aircraft.

Any outdoor (and many indoor) recordings contain faint, usually inaudible
tones from aircraft flying over: the propeller blade-pass fundamental and its
harmonics, between roughly 60 and 400 Hz. As the aircraft passes, the tone
sweeps down in frequency. The exact shape of that sweep depends on where the
listener was relative to the aircraft's real trajectory. With historical
ADS-B data for every flight that day, each candidate location on the ground
predicts a different curve, and the recording picks out the one that matches.

This is a reimplementation of the technique shown in colsto's video
["I discovered a sound that leaks your exact location"](https://www.youtube.com/watch?v=UytYrhYLOKI).
No code was released with that video; this project is an open, automatic
version of the method. See `docs/superpowers/specs/` for the design.

## Status

Proof of concept, validated on two recordings with known answers. On the
author's reference recording (Oyster Bay, NY, 2026-09-08) a blind search
over a 100 x 150 km region ranks the true location first, 43 m off after
refinement. On a 12-hour public eagle-cam stream (North Fort Myers, FL,
2024-12-28), with the flyovers found automatically, the search ranks the
right pair of flights first of 607 and places the camera within 600 m.
See Results below.

## Pipeline

1. `spectrogram` — decode any audio/video with ffmpeg, high-resolution STFT of
   the 40–450 Hz band, whitened per frequency bin so only *changing* tones stand out.
2. `detect` — find the flyovers without a person: slide a window over the
   whole recording, grid-search an ideal straight-pass Doppler curve
   (f0, speed, CPA slant distance, CPA time) in each, keep the local maxima,
   refine them and rank them. `fit` does the same for one window you choose.
   `detect` scores a sweep-weighted ridge contrast, which ignores bursts and
   steady tones but gives the flat ends of a curve little weight; `fit` scores
   the plain mean brightness under the curve, as the results below were fitted
   (`--objective contrast` switches it to the detector's score).
   The curve parameters only seed a search with coarse ranges.
3. `fetch-day` — stream one day's [adsb.lol](https://github.com/adsblol/globe_history_2026)
   archive straight from GitHub (4 GB, nothing written to disk) and keep the
   flights that touched a bounding box, dropping known jets using ICAO Doc
   8643 engine classes. `extract-region` does the same from an archive you
   already have on disk.
4. `search` — for every grid point (200 m) and every flight, compute the
   closest point of approach (CPA), its slant distance (using a DEM for the
   observer's elevation) and the *heard* CPA time (propagation delay). Keep
   points that hear an event-1-like pass, then exactly Δt later an
   event-2-like pass, with no other prop flight in between. Model the exact
   Doppler curve each survivor would hear from the real trajectory
   (emission-time radial velocity) and score it against the spectrogram.
   Event 1's placement absorbs measurement error; event 2's placement is
   locked to the flight geometry. `--score ridge` scores how closely the
   modelled curve follows the ridge traced in the window instead of the
   brightness under it (see "Ridge score" below); `--v-tol` and `--d-tol`
   open the speed and distance gates when the fit of a faint or turning
   pass cannot be trusted.
5. `refine` — 50 m grid around a winning candidate. Further detected
   flyovers can be added here: each is matched to a flight once, at the
   candidate, then scored at every fine point with the same time shift as
   event 1, so every event is locked to one clock.

## Usage

```bash
uv sync
uv run dopplertrack spectrogram recording.mp4 rec.npz
uv run dopplertrack detect rec.npz --band 60 300 --top 10 --out detected.json
uv run dopplertrack fit rec.npz --window 2090 2200 --band 95 200 --harmonics 2 --out ev1.json
uv run dopplertrack fit rec.npz --window 3020 3130 --band 95 195 --out ev2.json
uv run dopplertrack fetch-day 2026-09-08 traces_ny.npz --bbox 40.4 41.3 -74.4 -72.6
uv run dopplertrack search rec.npz --events ev1.json ev2.json --traces traces_ny.npz \
    --bbox 40.4 41.3 -74.4 -72.6 --spacing 200 --harmonics 2 1 --out results.csv
uv run dopplertrack refine rec.npz results.csv --events ev1.json ev2.json --traces traces_ny.npz \
    --bbox 40.4 41.3 -74.4 -72.6 --harmonics 2 1 --extra ev3.json ev4.json --out fine.csv
# one faint flyover whose heard-CPA time is known (here 20:40:58 UTC), gates opened wide
uv run dopplertrack fit rec.npz --window 1375 1465 --band 70 105 --harmonics 2 --out ev_faint.json
uv run dopplertrack search-single rec.npz --event ev_faint.json --traces traces_ny.npz --utc 1788900058.6 \
    --bbox 40.4 41.3 -74.4 -72.6 --harmonics 2 --v-tol 1.0 --d-tol 0.05 20 --out single.csv
```

More commands, each covering a step the video does by hand or discusses:

```bash
# the video's "Building the Doppler Curve, Step by Step" panel for one flight, observer and time
uv run dopplertrack explain --traces traces_ny.npz --icao ace391 --point 40.876466 -73.524951 \
    --utc 2026-09-08T21:08:43 --f0 104 --plot panel.png
# score a faint curve by its distance from a traced line, as the video does
uv run dopplertrack trace-line rec.npz ev_faint.json              # automatic ridge, reliable frames only
uv run dopplertrack trace-line rec.npz ev_faint.json --csv my_trace.csv --harmonic 2   # or a hand-traced line
uv run dopplertrack search-single ... --score line
# the filter funnel the video reports, weather, turn-aware trajectories, baro datum
uv run dopplertrack search ... --funnel --temp 24 --wind 4 200 --interp hermite --calibrate-baro
# look at the answer: Street View links for every candidate, and the figures
uv run dopplertrack export results.csv candidates.html --truth 40.876466 -73.524951
uv run dopplertrack plot pairs pairs.png --results results.csv
uv run dopplertrack plot candidate cand.png --spec rec.npz --results results.csv --events ev1.json ev2.json \
    --traces traces_ny.npz --bbox 40.4 41.3 -74.4 -72.6 --harmonics 2 1
# countermeasures: does a recording leak, and remove the leak
uv run dopplertrack audit rec.npz --band 60 300 --out leaks.json
uv run dopplertrack scrub recording.mp4 clean.wav --events leaks.json --band 60 300
```

`search --point LAT LON` scores a single known location instead of a grid,
which is the quickest way to validate the data and physics on a recording
whose location you already know. `detect` prints a ranked table of
candidate flyovers; `--out` writes them as a JSON list whose entries can be
passed to `search --events` as they are.

## Results

Reference recording: colsto's raw one-hour clip
([youtu.be/4CxVxuMVkMc](https://youtu.be/4CxVxuMVkMc)), made 2026-09-08 in
Oyster Bay, NY at 40.876466, -73.524951, the answer given in the video. Only
the recording date and the search region were used as inputs; the clip's
start time (20:17:30 UTC) and the two aircraft fall out of the search.

**Known-point check.** Scoring the true location against every non-jet flight
of the day (1,016 traces touching the region) leaves exactly one pair that
passes the 918 s offset and silence filters: N22NY, a Pilatus PC-12 at 6.1 km,
heard CPA 20:53:25 UTC, and N93CQ, a King Air 350 at 3.6 km, 21:08:43 UTC.
The Doppler curves modelled from their real trajectories lie on the measured
ridges with no free parameters beyond a half-second time shift:

![Modelled curves at the true point](docs/results/oyster_bay_known_point.png)

**Blind search.** A 200 m grid over a 100 x 150 km box around New York
(379,758 points, 3,068 airborne flight segments) gives 86 million
(point, flight) closest approaches and 40,889 candidate (point, pair)
triples after the filters, from 328 distinct flight pairs. Scoring all of them
takes 19 minutes on a laptop.

| | |
|---|---|
| Rank of the true location's flight pair | 1 of 328 |
| Top candidate, 200 m grid | 123 m from truth |
| Refined peak, 50 m grid | 43 m from truth |
| Margin over the best wrong flight pair | 4.9 % (27.1 vs 25.8) |

![Blind search score map](docs/results/oyster_bay_search.png)

The result is right and the 50 m refinement shows a single clean peak. The
4.9 % margin over the runner-up pair (flights 70 km away whose geometry
happens to produce a similar 918 s offset) is thin, because the score only
measures brightness under the modelled curve, so a wrong pair can score well
wherever the spectrogram is busy. Numbers:
`docs/results/oyster_bay_search_best_per_pair.csv` (best score per flight
pair) and `docs/results/oyster_bay_refined_50m.csv`.

**Automatic event detection.** The two windows above were chosen by eye, as
in the video. Running `detect` over the whole hour with no hints returns the
same two flyovers as its top two candidates (CPA 2155.6 s and 3068.4 s,
scores 15.7 and 10.0; the third candidate scores 9.0). Its own fits of them
can't stand in for `fit` yet: its event 2 comes out with f0 11 % above what
the King Air's trajectory needs and the time between the two events 10 s
short, so the search's ±6 s and ±5 % filters reject the true pair at the true
location. Finding the flyovers needs no manual input; measuring them for the
search still uses the two `fit` windows.

### Second recording: a 12-hour eagle cam

To test on something the method was not developed on, we used a public
livestream archive with a published location: the Southwest Florida Eagle
Cam in North Fort Myers (nest LE-026B at 26.702359, -81.847766 in Lee
County's eagle-nest register), the 12-hour archive of 2024-12-28
([youtu.be/zs2o5Rhlp5w](https://youtu.be/zs2o5Rhlp5w)). It is a harder
recording: an outdoor microphone in wind, 12 hours long, and 10 km from a
busy flight-school airport, so prop aircraft pass every few minutes. The
day's ADS-B was streamed with `fetch-day` (557 non-jet flights touching a
130 x 130 km box, 4 GB read, nothing stored) and the flyovers were found
with `detect`: of its top 40 candidates, 20 coincide within 25 s with a
known prop flight passing within 12 km of the nest.

![Known-point check on the eagle cam](docs/results/swfl_known_point.png)

**Blind search** on two detected events 338 s apart (a Beech Baron at
2.5 km and a Pilatus PC-12 at 6 km, as it turns out) over the 130 x 130 km
box: 432,196 points, 1,233 flight segments, 50,682 candidate triples from
607 flight pairs, 8 minutes.

| | |
|---|---|
| Rank of the true flight pair | 1 of 607 |
| Margin over the best wrong pair | 2.5 % |
| Top candidate, 200 m grid, two events | 917 m from the nest |
| Refined peak, 50 m grid, two events | 1.7 km from the nest |
| Refined peak with three more detected flyovers | 592 m from the nest |

![Eagle cam search](docs/results/swfl_search.png)

The right flights are identified, but two events pin the location only to
about 2 km here, and the reason is geometry rather than noise: both
aircraft passed nearly overhead at 2.5 km and 5.7 km altitude, so the slant
distance that shapes the Doppler curve barely changes as the observer moves
a kilometre on the ground. At Oyster Bay the King Air passed 3.5 km to the
side at 1 km altitude, which is why that peak is sharp. Adding three more
flyovers found by `detect` (`refine --extra`), locked to the same clock,
brings the peak to 592 m. Lower, lateral passes are what pin a location,
so the more flyovers a long recording offers, the better. Numbers:
`docs/results/swfl_search_best_per_pair.csv`, `docs/results/swfl_refined_50m.csv`,
and the detector's candidates for both recordings in `docs/results/*_detected.json`.

**What this second recording changed.** The first version of the scorer
averaged brightness only over the part of the window a modelled curve
covered, so a flight that landed or left the data mid-pass was judged on its
bright half alone; on the eagle cam that put a pair 64 km away on top. The
score is now the brightness summed along the curve over the whole measured
window (uncovered parts count as zero), which fixed the ranking here and
left the Oyster Bay result unchanged. Scoring the curve against its
spectral neighbourhood (a contrastive score) was tried as well: it widens
the Oyster Bay margin but does not help the eagle cam, so it is not in.

### A faint single flyover with a known time

The video's last experiment locates the recording from one faint one-minute
curve, given its time. The Oyster Bay hour has such a curve at 1375–1465 s:
a fundamental sweeping 92 to 73 Hz a few dB above the noise, with its second
harmonic ([`docs/results/oyster_bay_ev_faint.json`](docs/results/oyster_bay_ev_faint.json)).
The only ADS-B flight passing the true point then is N7681Y, a Piper Twin
Comanche, whose heard CPA at the true point is 0.9 s from the fitted one.
Its curve modelled from the real trajectory sits on the faint ridge
([figure](docs/results/oyster_bay_faint_curve.png)), but the straight-pass
fit of that ridge says 41 m/s at 750 m where the aircraft was doing 70 m/s
at 8.2 km: a sharp drop followed by a flat tail reads as a close slow pass,
and the search's default gates (±35 % in speed, 0.4–2.5× in distance) drop
the true flight before scoring. With the gates opened (`--v-tol 1.0
--d-tol 0.05 20`), `search-single` over the same 100 x 150 km box and ±30 s
scores 77,280 (point, flight) pairs from 66 flights:

| score | top point | margin over the next flight | top-20 points on N7681Y | of those within 2 km |
|---|---|---|---|---|
| brightness | 1.7 km from the truth, N7681Y | 31.6 % | 20 | 17 |
| ridge | 2.3 km from the truth, N7681Y | 14.5 % | 20 | 3 |

Tables: [`oyster_bay_single_faint_brightness.csv`](docs/results/oyster_bay_single_faint_brightness.csv),
[`oyster_bay_single_faint_ridge.csv`](docs/results/oyster_bay_single_faint_ridge.csv).
Two lessons. A single faint flyover with a known time does locate the
recording to a couple of kilometres in a box this size, as the video says.
And the ideal-curve fit is the weak link: on a pass that is not straight it
is wrong by a factor of ten in distance, so the gates it seeds must be loose
or, better, replaced by the trajectory-modelled curve itself.

**Ridge score.** For that faint curve the video scores candidates by their
distance from a hand-traced line rather than by brightness. `--score ridge`
is the automatic version: a Viterbi ridge is traced through the event
window once (harmonics are multiples of it), frames are kept only where the
ridge is bright and at least 3 dB brighter than the best path that avoids
it (a bird burst or a second tone offers several equally good paths and says
nothing), and a modelled curve scores the fraction of kept frames it passes
within 1.5 % in frequency of ([tracks and masks](docs/results/ridge_tracks.png)).
On the two blind two-event searches it is not an improvement over
brightness, and it is not the default:

| recording | brightness: true pair rank, margin | ridge: rank, margin |
|---|---|---|
| Oyster Bay (328 pairs) | 1, 4.9 % | 1, 4.0 % |
| eagle cam (607 pairs) | 1, 2.5 % | 4 (1 without the uniqueness mask, 0.6 %) |

Widening the tolerance to 3 % or 5 % or dropping the mask moves the eagle
cam between rank 1 with no margin and rank 4; the pair it loses to is the
one brightness ranks second, two aircraft on the same airway whose curves
differ little. Best-per-pair tables under both scores:
[`oyster_bay_search_best_per_pair_ridge.csv`](docs/results/oyster_bay_search_best_per_pair_ridge.csv),
[`swfl_search_best_per_pair_ridge.csv`](docs/results/swfl_search_best_per_pair_ridge.csv).
What the ridge score does give is a score in [0, 1] with a meaning (the
share of the measured curve the candidate explains) and a per-frame
residual, which is the raw material for an uncertainty estimate.

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
| 5 | Historical ADS-B (a paid Flightradar24 API), jets dropped | `fetch-day` (free adsb.lol), ICAO Doc 8643 engine classes |
| 6 | For every flight, a 200 m grid over a corridor around its track: CPA and the sound-delayed heard CPA time | `search` (`cpa_table`, KD-tree corridor) |
| 7 | Filters: the offset between the two events (920 s), speed and distance, then silence (no other plane heard in between); the video reports how many points survive each | `pair_candidates`; `search --funnel` prints the same funnel |
| 8 | Survivors grouped by flight pair; exact curves from the real, turning track; event 2 locked to event 1's clock; one pair scores far above the rest | locked scoring; `plot pairs` |
| 9 | Turns, speed and altitude changes are encoded in the curve and break ambiguities | real trajectories; `--interp hermite` makes turns between ADS-B reports curve (see below) |
| 10 | The 200 m grid is "a little too coarse": the right answer can be 141 m off and score badly | `refine` (50 m) |
| 11 | Street View at the best point shows the camera's surroundings | `export` (GeoJSON, KML, HTML map, each point with a Street View link) |
| 12 | A second example: one faint ~60 s flyover with a known time, scored by distance from a traced line | `search-single`; `--score line` with `trace-line` (hand-traced CSV or the automatic ridge) |
| 13 | Countermeasures: a high-pass filter does not help (boosting brings it back); talking over it does not; it survives phone-call compression | `audit`, `scrub`; the claims are checked in `tests/test_countermeasures.py` |
| 14 | Still air, c = 343 m/s | `--temp`, `--wind`, `--weather` |

### Added in this round, and what is and is not validated

None of these changes the default behaviour, so every number in Results above is still what
the defaults give. They are tested on synthetic data with known answers; none has yet been
run on the two real recordings (the recordings are YouTube uploads and could not be
downloaded where this was built), so on real data they are options to try, not defaults.

- **`explain`.** The video's step-by-step panel for any flight, observer and reception time,
  including what the reception-time model would say. On the video's own numbers it gives
  t_e = 16.910 s, r = 2775 m, v_r = 121.16 m/s and f = 96.07 Hz.
- **Filter funnel** (`search --funnel`): points, (point, flight, flight) triples and flight
  pairs after the offset filter alone, after speed and distance, and after silence, the
  counts the video quotes. The offset-only stage is the expensive one, so it is opt-in.
- **Recording clock.** Every search and refine row now has `rec_start_unix`, the UTC time of
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
  so the gain on real data is smaller than this and has to be measured.
- **Baro datum** (`--calibrate-baro`). Geometric altitude is GNSS height; barometric altitude
  assumes a standard atmosphere and is off by tens of metres on any non-standard day (76 m on
  the test trace, QNH 1026 hPa). Aircraft that report both measure the offset; it is fitted per
  half hour as a + b x baro and applied to aircraft that only report baro.
- **Weather** (`--temp`, `--wind SPEED FROM_DEG`, `--weather`). The speed of sound follows
  temperature (0.6 m/s per degree C). A uniform wind is modelled exactly: the sound heard at
  t_e + tau left when |D + w tau| = c tau (D from observer to aircraft), which gives
  f = f0 (c - m.w) / (c - m.w + m.v) with m the unit vector along that ray. The obvious shortcut
  (a direction-dependent sound speed c + w.k with the ray straight) is off by about as much
  as ignoring the wind, because the ray tilts by about |w|/c and that tilt multiplies the
  aircraft's speed; the test checks the model against an exact air-frame ray solution. `--weather` reads
  ERA5 surface temperature and wind from Open-Meteo's free archive and carries them to the
  middle of the sound path (standard lapse rate, 1/7 power-law wind profile).
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
- **Several trace files** (`--traces day1.npz day2.npz`), for a recording whose UTC day is
  uncertain or that spans midnight UTC.

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

## Changes from the method in the video

The physics and the search strategy are the author's. What differs:

- **Automatic curve measurement.** The video measures each Doppler curve by
  hand, sliding f0 / speed / distance / CPA-time until the overlay looks
  right, then choosing "generous ranges" by eye. `fit` does this without a
  person: the window is whitened per frequency bin so stationary tones
  (mains hum, machinery) vanish, a vectorised grid search over all four
  parameters maximises spectrogram brightness under the curve, and a
  Nelder-Mead pass refines it. The ideal-curve fit still only seeds coarse
  filters; final ranking uses real trajectories.
- **Harmonic-aware fitting and scoring.** When a flyover shows a second
  harmonic, fundamental and harmonic are scored together, which constrains
  the curve more tightly than one line.
- **Open, documented data path.** The video used a paid/scraped ADS-B feed
  and did not release code. Here every input is a public dataset with a
  loader: adsb.lol daily archives, ICAO Doc 8643 engine classes, AWS
  terrain tiles.
- **Search engineering.** A KD-tree corridor around each flight path limits
  CPA computation to grid points that could possibly hear it; distance math
  is chunked so multi-hour tracks do not exhaust memory; the time-shift and
  f0 sweeps of the locked scorer run as one array operation per candidate.
- **Single-event, known-time mode as a first-class command.** The video
  demonstrates locating from one faint one-minute curve when the recording
  time is known; `search-single` exposes that directly, and reproduces it on
  the Oyster Bay hour (see above).
- **Ridge scoring as an option.** The video scores its faint curve by
  distance from a hand-traced line; `--score ridge` traces the line itself,
  with a uniqueness mask instead of a person's judgement of where the line
  is real.

Kept unchanged on purpose: emission-time radial velocity (not
observation-time), propagation-delayed CPA times, slant distance with a DEM,
the inter-event offset filter, the silence filter, and locked two-event
scoring where event 1 absorbs measurement error and event 2 is placed purely
by flight geometry.

## Data sources (all free)

| Need | Source |
|---|---|
| Historical ADS-B | adsb.lol `globe_history_YYYY` GitHub releases (~4 GB/day) |
| Aircraft engine class | OpenSky mirror of ICAO Doc 8643 |
| Elevation | AWS Terrain Tiles (Terrarium) |
| Weather (optional) | Open-Meteo ERA5 archive |

## Development

```bash
uv run pytest
```

## Releasing

Releases are driven by git tags. Pushing a tag `vX.Y.Z` that matches the
version in `pyproject.toml` runs `.github/workflows/release.yml`, which runs
the tests, builds the sdist and wheel, attaches them to a GitHub Release with
generated notes, and publishes to PyPI through trusted publishing (no API
token stored anywhere).

```bash
uv version 0.2.0            # bump pyproject.toml
git commit -am "Release 0.2.0"
git tag v0.2.0 && git push && git push --tags
```

One-time setup for PyPI: on pypi.org, add a *pending publisher* for project
`dopplertrack` with owner `mjshiggins`, repository `dopplertrack`, workflow
`release.yml`, environment `pypi`, and create a `pypi` environment in the
GitHub repository settings. Until that exists the `pypi` job fails and the
GitHub Release still succeeds.
