"""
Layer 1: rule-based validation checks.

Every rule returns issues as dicts:
  {observation_id, layer, rule, severity, message, volunteer_question}
Severity: "error" (almost certainly wrong) or "warning" (needs a human look).
"""
from __future__ import annotations

import math
from datetime import datetime

import pandas as pd

from truestream.generate_data import SITES

REQUIRED = ["site_id", "timestamp", "latitude", "longitude", "flow", "health_rating"]
MAX_DISTANCE_KM = 1.0
DUPLICATE_WINDOW_MIN = 10


def _issue(obs_id, rule, severity, message, question=""):
    return {
        "observation_id": obs_id,
        "layer": "rules",
        "rule": rule,
        "severity": severity,
        "message": message,
        "volunteer_question": question,
    }


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def check_required(row) -> list[dict]:
    missing = [f for f in REQUIRED if pd.isna(row.get(f))]
    if missing:
        return [_issue(row.observation_id, "missing_field", "error",
                       f"Required field(s) missing: {', '.join(missing)}.",
                       f"Could you fill in: {', '.join(missing)}?")]
    return []


def check_temperature(row) -> list[dict]:
    t = row.get("water_temp_c")
    if pd.isna(t):
        return []
    if 41 <= t <= 95:
        c = round((t - 32) * 5 / 9, 1)
        return [_issue(row.observation_id, "temp_fahrenheit", "warning",
                       f"Water temperature {t} °C is far too hot for a stream, but equals {c} °C "
                       "if it was entered in Fahrenheit.",
                       f"Did you measure {t} °F? That would be {c} °C.")]
    if t < -1 or t > 40:
        return [_issue(row.observation_id, "temp_impossible", "error",
                       f"Water temperature {t} °C is physically implausible for liquid stream water.",
                       "Could you re-check the thermometer reading?")]
    return []


def check_ph(row) -> list[dict]:
    ph = row.get("ph")
    if pd.isna(ph):
        return []
    if ph > 14 or ph < 0:
        if 40 <= ph <= 100:
            return [_issue(row.observation_id, "ph_missing_decimal", "warning",
                           f"pH {ph} is outside the 0–14 scale; {ph / 10:.1f} looks like the "
                           "intended value (missing decimal point).",
                           f"Did you mean pH {ph / 10:.1f}?")]
        return [_issue(row.observation_id, "ph_impossible", "error",
                       f"pH {ph} is outside the possible 0–14 scale.",
                       "Could you re-check the pH strip or meter?")]
    if ph < 4 or ph > 10:
        return [_issue(row.observation_id, "ph_unusual", "warning",
                       f"pH {ph} is very unusual for a stream and may signal pollution or an error.",
                       "Could you confirm the pH reading?")]
    return []


def check_dry_contradiction(row) -> list[dict]:
    if row.get("flow") != "dry":
        return []
    measured = [f for f in ("water_temp_c", "ph") if not pd.isna(row.get(f))]
    if measured:
        return [_issue(row.observation_id, "dry_but_measured", "error",
                       f"Stream marked as dry, but water measurements were recorded ({', '.join(measured)}).",
                       "Was the stream completely dry, or was there some water?")]
    return []


def check_timestamp(row, now: datetime) -> list[dict]:
    ts = row.get("timestamp")
    if pd.isna(ts):
        return []
    if pd.Timestamp(ts) > pd.Timestamp(now):
        return [_issue(row.observation_id, "future_timestamp", "error",
                       f"Observation date {pd.Timestamp(ts):%Y-%m-%d} is in the future.",
                       "Could you check the date on your device?")]
    return []


def check_location(row) -> list[dict]:
    site = SITES.get(row.get("site_id"))
    lat, lon = row.get("latitude"), row.get("longitude")
    if site is None or pd.isna(lat) or pd.isna(lon):
        return []
    _, slat, slon = site
    dist = haversine_km(lat, lon, slat, slon)
    if dist <= MAX_DISTANCE_KM:
        return []
    if haversine_km(lon, lat, slat, slon) <= MAX_DISTANCE_KM:
        return [_issue(row.observation_id, "gps_swapped", "warning",
                       "Latitude and longitude appear to be swapped; swapping them puts the point at the site.",
                       "")]
    return [_issue(row.observation_id, "gps_far", "warning",
                   f"Location is {dist:.1f} km from the registered site '{site[0]}'.",
                   "Were you at a different spot on the stream? Please confirm the site.")]


def check_odor_rating(row) -> list[dict]:
    if row.get("odor") in ("sewage", "chemical") and row.get("health_rating", 0) >= 4:
        return [_issue(row.observation_id, "odor_rating_conflict", "warning",
                       f"Odor recorded as '{row.odor}' but stream health rated {int(row.health_rating)}/5.",
                       "You noticed a strong smell. Would you like to review the health rating?")]
    return []


def check_insects(row) -> list[dict]:
    n = row.get("insects_count")
    if pd.isna(n):
        return []
    if n < 0 or n > 500:
        return [_issue(row.observation_id, "insects_implausible", "warning",
                       f"Insect count {int(n)} is outside the plausible range (0–500).",
                       "Could you re-check the insect count?")]
    return []


def check_duplicates(df: pd.DataFrame) -> list[dict]:
    issues = []
    d = df.dropna(subset=["volunteer_id", "site_id", "timestamp"]).sort_values("timestamp")
    for _, grp in d.groupby(["volunteer_id", "site_id"]):
        prev = None
        for _, row in grp.iterrows():
            if prev is not None:
                gap = (row.timestamp - prev.timestamp).total_seconds() / 60
                if gap <= DUPLICATE_WINDOW_MIN:
                    issues.append(_issue(row.observation_id, "duplicate", "warning",
                                         f"Looks like a duplicate of {prev.observation_id} "
                                         f"(same volunteer and site, {gap:.0f} min apart).",
                                         "Did you submit this observation twice?"))
            prev = row
    return issues


ROW_RULES = [check_required, check_temperature, check_ph, check_dry_contradiction,
             check_location, check_odor_rating, check_insects]


def run_rules(df: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    now = now or datetime.now()
    issues: list[dict] = []
    for row in df.itertuples(index=False):
        row = pd.Series(row._asdict())
        for rule in ROW_RULES:
            issues += rule(row)
        issues += check_timestamp(row, now)
    issues += check_duplicates(df)
    cols = ["observation_id", "layer", "rule", "severity", "message", "volunteer_question"]
    return pd.DataFrame(issues, columns=cols)
