"""Hand candidates to a person: GeoJSON, KML and a standalone HTML map, each point with
Google Maps and Street View links.

The video's last step is visual: with the best point found, Street View at that spot shows
the tree line and the pole the camera hangs on. These exports make that step one click for
every candidate, the best point of each flight pair by default."""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone

import pandas as pd


def maps_url(lat: float, lon: float) -> str:
    return f"https://www.google.com/maps/search/?api=1&query={lat:.6f},{lon:.6f}"


def streetview_url(lat: float, lon: float) -> str:
    return f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat:.6f},{lon:.6f}"


def candidates(res: pd.DataFrame, top: int = 20, per_pair: bool = True) -> pd.DataFrame:
    """Best rows, highest score first; per_pair keeps only the best point of each flight pair
    (search results) or of each flight (search-single results)."""
    r = res.sort_values("score", ascending=False)
    if per_pair:
        key = [k for k in (("icao1", "icao2"), ("icao",)) if all(c in r.columns for c in k)]
        if key:
            r = r.drop_duplicates(list(key[0]))
    r = r.head(top).reset_index(drop=True)
    r.insert(0, "rank", range(1, len(r) + 1))
    return r


def _props(row: pd.Series) -> dict:
    out = {}
    for k, v in row.items():
        if k in ("lat", "lon"):
            continue
        if hasattr(v, "item"):
            v = v.item()
        if isinstance(v, float) and v != v:
            v = None
        out[k] = v
    if "rec_start_unix" in row and row["rec_start_unix"] == row["rec_start_unix"]:
        out["rec_start_utc"] = datetime.fromtimestamp(float(row["rec_start_unix"]), timezone.utc).isoformat()
    out["maps"] = maps_url(row.lat, row.lon)
    out["streetview"] = streetview_url(row.lat, row.lon)
    return out


def to_geojson(cands: pd.DataFrame) -> str:
    feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [float(r.lon), float(r.lat)]},
              "properties": _props(r)} for _, r in cands.iterrows()]
    return json.dumps({"type": "FeatureCollection", "features": feats}, indent=1)


def _label(r: pd.Series) -> str:
    who = f"{r.icao1}/{r.icao2}" if "icao1" in r else (str(r.icao) if "icao" in r else "")
    return f"#{int(r['rank'])} {who} score {r.score:.3f}".strip()


def to_kml(cands: pd.DataFrame) -> str:
    pm = []
    for _, r in cands.iterrows():
        p = _props(r)
        desc = "<br/>".join(f"{html.escape(str(k))}: {html.escape(str(v))}" for k, v in p.items() if k not in ("maps", "streetview"))
        desc += f'<br/><a href="{p["streetview"]}">Street View</a> <a href="{p["maps"]}">Map</a>'
        pm.append(f"<Placemark><name>{html.escape(_label(r))}</name><description><![CDATA[{desc}]]></description>"
                  f"<Point><coordinates>{r.lon:.6f},{r.lat:.6f},0</coordinates></Point></Placemark>")
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
            "<name>dopplertrack candidates</name>" + "".join(pm) + "</Document></kml>\n")


def to_html(cands: pd.DataFrame, truth: tuple[float, float] | None = None) -> str:
    """A standalone page: Leaflet map (OpenStreetMap tiles) plus a ranked table."""
    pts = [{"lat": float(r.lat), "lon": float(r.lon), "label": _label(r), "sv": streetview_url(r.lat, r.lon),
            "maps": maps_url(r.lat, r.lon)} for _, r in cands.iterrows()]
    rows = "".join(f"<tr><td>{html.escape(p['label'])}</td><td>{p['lat']:.6f}, {p['lon']:.6f}</td>"
                   f"<td><a href='{p['sv']}' target='_blank'>Street View</a> · <a href='{p['maps']}' target='_blank'>Map</a></td></tr>"
                   for p in pts)
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>dopplertrack candidates</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>body{{font:14px system-ui,sans-serif;margin:0 16px}}#m{{height:60vh;margin:12px 0}}table{{border-collapse:collapse;width:100%}}td{{padding:4px 8px;border-bottom:1px solid #ddd}}</style>
</head><body><h2>dopplertrack candidates</h2><div id="m"></div><table>{rows}</table>
<script>
const pts={json.dumps(pts)}, truth={json.dumps(truth)};
const m=L.map('m'); L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{maxZoom:19,attribution:'© OpenStreetMap'}}).addTo(m);
const b=[]; pts.forEach((p,i)=>{{L.circleMarker([p.lat,p.lon],{{radius:i?5:9,color:i?'#3366cc':'#d62728'}}).addTo(m)
 .bindPopup(p.label+'<br><a target=_blank href="'+p.sv+'">Street View</a>'); b.push([p.lat,p.lon]);}});
if(truth){{L.marker(truth).addTo(m).bindPopup('truth'); b.push(truth);}}
if(b.length) m.fitBounds(b,{{padding:[30,30],maxZoom:15}});
</script></body></html>
"""
