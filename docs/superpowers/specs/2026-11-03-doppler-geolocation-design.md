# Doppler-curve acoustic geolocation — proof-of-concept design

Date: 2026-11-03

## Goal

Reproduce the technique from colsto's video *"I discovered a sound that leaks
your exact location"* (youtube.com/watch?v=UytYrhYLOKI): locate where a
recording was made by matching the (usually inaudible) Doppler curves of
propeller-aircraft flyovers in the recording's 50–400 Hz band against
historical ADS-B flight tracks.

Phase 1 (this spec): a proof-of-concept pipeline, validated on the author's
own reference recording (youtu.be/4CxVxuMVkMc) whose answer is known:
(40.876466, -73.524951), Oyster Bay NY, recorded 2026-09-08.
Phase 2: package it as an open-source library. Phase 3: a visual front end.

## Physics

An aircraft with a tonal source (propeller blade-pass fundamental f0 and its
harmonics) heard by a stationary observer produces

    f_obs(t) = f0 · c / (c + v_r(t_e))

where c is the speed of sound, v_r is the radial velocity (range rate) at the
*emission* time t_e, and the observation time is t = t_e + r(t_e)/c.
Modelling with the radial velocity at observation time instead of emission
time gives a visibly wrong curve, not just a delayed one.

For a straight, constant-speed pass the curve is fully determined by
(f0, v, d_cpa, t_cpa): v sets the size of the frequency drop and d_cpa sets
how fast the drop happens. Real curves deviate because aircraft turn,
climb and change speed, and that deviation is what makes a candidate
location uniquely scoreable.

## Data sources (all free)

| Need | Source |
|---|---|
| Historical ADS-B, whole day, worldwide | adsb.lol `globe_history_YYYY` GitHub releases, one ~4 GB tar per day of readsb `trace_full_<hex>.json` (gzip) files |
| Aircraft type → engine class | OpenSky mirror of ICAO Doc 8643 (`doc8643AircraftTypes.csv`): Piston / Turboprop / Jet |
| Elevation (DEM) | AWS Terrain Tiles (`elevation-tiles-prod`, Terrarium PNG), zoom 11–12 |
| Audio | any file via ffmpeg → mono 4 kHz WAV |

## Approaches considered

1. **Faithful reimplementation of the author's pipeline** (grid of
   candidate observers, CPA-time filters, physics-modelled curves scored
   against the spectrogram with locked two-event scoring). Known to work;
   computationally heavy but tractable with pre-filters. **Chosen.**
2. Direct inversion: fit (v, d, t0, f0) per curve and solve for flight
   geometry. Rejected: aircraft don't fly straight, so the ideal-curve fit
   is only good for coarse filtering.
3. Brute-force score every grid point × every flight with no pre-filters.
   Simple but O(points × flights × samples); fine only for small regions.
   Used as the validation mode (score the known location directly).

## Pipeline

```
audio ──► spectrogram (STFT, dB, per-bin median whitening)
      ──► event extraction (ridge in a user-chosen window; ideal-curve fit → v, d, t_cpa, f0 ± ranges)
ADS-B ──► prop-aircraft traces in region/day (resampled to 1 s, ENU)
DEM   ──► ground elevation per candidate
      ──► candidate observers: for each prop flight segment matching an event's
          (v, d) range, points at the matching slant distance from the track
      ──► pairing: candidate hears event-1-like flight, then event-2-like flight
          exactly Δt later (± a few s), with no other prop CPA in between (silence)
      ──► scoring: model the exact Doppler curve each candidate would hear
          (emission-time radial velocity, propagation delay) for each event and
          sum whitened spectrogram brightness under the curve; sweep f0 and
          t_cpa for event 1, lock event 2's placement to flight geometry
      ──► ranked candidates + per-flight-pair best point + heat map
```

## Modules (`src/dopplertrack/`)

- `audio.py` — decode, STFT, whitening → `Spectrogram(t, f, S)`.
- `doppler.py` — ideal curve, curve from a real trajectory, speed of sound.
- `geo.py` — WGS-84 → local ENU, slant range, haversine helpers.
- `tracks.py` — ridge extraction and ideal-curve fitting for an event window.
- `adsb.py` — read adsb.lol archive, type lookup, resample traces.
- `terrain.py` — cached Terrarium tile elevation lookup.
- `search.py` — candidate generation, pairing, silence filter.
- `score.py` — curve placement and locked two-event scoring.
- `geoid.py` — EGM96 undulation, so terrain (MSL) and ADS-B GNSS altitude (ellipsoid) share a datum (added later).
- `weather.py` — sound speed from temperature, wind, optional ERA5 lookup (added later).
- `privacy.py` — `audit` (detected flyovers against a time-reversed chance level) and `scrub` (added later).
- `export.py` — GeoJSON / KML / HTML candidates with Street View links (added later).
- `cli.py` — `dopplertrack spectrogram|fit|detect|search|refine|search-single|explain|trace-line|audit|scrub|export|plot`.

## Validation plan

1. Forward model at the true location for every prop flight on the day;
   the two flights whose modelled curves land on the observed ridges at
   2152 s and 3073 s of the clip must be found, confirming data + physics.
2. Blind search of a ~120 km box around New York with the event
   measurements only; the true location must rank first (or within the
   top few at 100 m) by combined score.
3. Stretch: widen to the state / northeast.

## Testing

Unit tests for the physics core (ideal curve symmetry and limits,
emission-time delay, ENU/slant range round trip, trajectory curve on a
synthetic straight pass matching the closed form). Pipeline stages are
exercised by the validation runs above.
