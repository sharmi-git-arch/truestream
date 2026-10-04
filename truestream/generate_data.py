"""
Generate realistic citizen stream-assessment records with deliberately planted errors.

Each record has two ground-truth columns used ONLY for the accuracy test:
  - planted_error       (True/False)
  - planted_error_type  (which mistake we planted, or "")

Adjust FIELD names to match the real OneAquaHealth app once you've seen it.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta

import pandas as pd

# Demo monitoring sites (invented names, coordinates are the registered site location).
SITES = {
    "S01": ("Willow Brook (upper)", 38.7605, -9.1702),
    "S02": ("Willow Brook (lower)", 38.7481, -9.1588),
    "S03": ("Mill Creek (park)", 38.7342, -9.1461),
    "S04": ("Mill Creek (bridge)", 38.7290, -9.1350),
    "S05": ("Heron Stream (school)", 38.7718, -9.1205),
    "S06": ("Heron Stream (outfall)", 38.7801, -9.1102),
}

CLARITY = ["clear", "slightly cloudy", "cloudy", "opaque"]
ODOR = ["none", "earthy", "sewage", "chemical", "other"]
FLOW = ["dry", "still", "slow", "moderate", "fast"]
TRASH = ["none", "low", "medium", "high"]

CLEAN_NOTES = [
    "Water looked normal, saw a few mayflies.",
    "Some leaves in the water, otherwise fine.",
    "Ducks near the bank. Light trash near the path.",
    "Water a bit muddy after yesterday's rain.",
    "Clear water, lots of caddisfly larvae under rocks.",
    "Quiet stream today, nothing unusual.",
    "Algae on some rocks near the edge.",
    "",
    "",
]

PLANTED_TYPES = [
    "missing_field",
    "temp_impossible",
    "temp_fahrenheit",
    "ph_missing_decimal",
    "dry_but_measured",
    "future_timestamp",
    "gps_swapped",
    "gps_far",
    "duplicate",
    "odor_rating_conflict",
    "notes_contradict_fields",  # subtle: only AI should catch this
    "junk_entry",               # subtle: only AI should catch this
]


def _clean_record(i: int, rng: random.Random, start: datetime) -> dict:
    site_id = rng.choice(list(SITES))
    _, lat, lon = SITES[site_id]
    flow = rng.choice(["still", "slow", "moderate", "moderate", "fast"])
    clarity = rng.choices(CLARITY, weights=[5, 3, 2, 0.5])[0]
    odor = rng.choices(ODOR, weights=[7, 3, 0.3, 0.2, 0.5])[0]
    trash = rng.choices(TRASH, weights=[4, 4, 2, 1])[0]
    rating = 5 if clarity == "clear" and trash in ("none", "low") else rng.choice([3, 4])
    if odor in ("sewage", "chemical"):
        rating = rng.choice([1, 2])
    return {
        "observation_id": f"OBS-{i:04d}",
        "volunteer_id": f"V{rng.randint(1, 25):02d}",
        "site_id": site_id,
        "site_name": SITES[site_id][0],
        "timestamp": start + timedelta(days=rng.randint(0, 60), hours=rng.randint(7, 18),
                                       minutes=rng.randint(0, 59)),
        "latitude": round(lat + rng.uniform(-0.0015, 0.0015), 6),
        "longitude": round(lon + rng.uniform(-0.0015, 0.0015), 6),
        "water_temp_c": round(rng.uniform(9, 21), 1),
        "ph": round(rng.uniform(6.6, 8.3), 1),
        "water_clarity": clarity,
        "odor": odor,
        "flow": flow,
        "trash_level": trash,
        "insects_count": rng.randint(0, 40),
        "health_rating": rating,
        "notes": rng.choice(CLEAN_NOTES),
        "planted_error": False,
        "planted_error_type": "",
    }


def _plant(rec: dict, kind: str, rng: random.Random, records: list[dict]) -> dict:
    rec = dict(rec)
    rec["planted_error"] = True
    rec["planted_error_type"] = kind
    if kind == "missing_field":
        rec[rng.choice(["site_id", "flow", "health_rating"])] = None
    elif kind == "temp_impossible":
        rec["water_temp_c"] = rng.choice([-12.0, 61.5, 150.0])
    elif kind == "temp_fahrenheit":
        rec["water_temp_c"] = round(rec["water_temp_c"] * 9 / 5 + 32, 1)
    elif kind == "ph_missing_decimal":
        rec["ph"] = round(rec["ph"] * 10)
    elif kind == "dry_but_measured":
        rec["flow"] = "dry"  # but temp / pH / clarity still recorded
    elif kind == "future_timestamp":
        rec["timestamp"] = datetime.now() + timedelta(days=rng.randint(20, 300))
    elif kind == "gps_swapped":
        rec["latitude"], rec["longitude"] = rec["longitude"], rec["latitude"]
    elif kind == "gps_far":
        rec["latitude"] += rng.choice([-1, 1]) * rng.uniform(0.05, 0.3)
    elif kind == "duplicate":
        src = rng.choice([r for r in records if not r["planted_error"]])
        rec = dict(src)
        rec["observation_id"] = f"{src['observation_id']}-DUP"
        rec["timestamp"] = src["timestamp"] + timedelta(minutes=rng.randint(1, 6))
        rec["planted_error"] = True
        rec["planted_error_type"] = kind
    elif kind == "odor_rating_conflict":
        rec["odor"] = "sewage"
        rec["health_rating"] = 5
    elif kind == "notes_contradict_fields":
        rec["odor"] = "none"
        rec["water_clarity"] = "clear"
        rec["health_rating"] = 5
        rec["notes"] = rng.choice([
            "Strong rotten-egg smell and grey foam near the pipe outlet.",
            "Several dead fish floating near the bank, water had an oily sheen.",
            "Water was brown and smelled like a toilet, kids were told not to touch it.",
        ])
    elif kind == "junk_entry":
        rec["notes"] = rng.choice(["test test 123", "asdfgh", "just trying the app lol"])
        rec["insects_count"] = rng.choice([999, 0])
    return rec


def generate(n_clean: int = 260, n_errors_each: int = 4, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    start = datetime.now() - timedelta(days=75)
    records = [_clean_record(i, rng, start) for i in range(1, n_clean + 1)]
    next_id = n_clean + 1
    for kind in PLANTED_TYPES:
        for _ in range(n_errors_each):
            base = _clean_record(next_id, rng, start)
            records.append(_plant(base, kind, rng, records))
            next_id += 1
    # Tricky-but-VALID records: they look suspicious but are correct.
    # A good system should NOT flag these (this measures false alarms).
    tricky = [
        {"odor": "sewage", "health_rating": 2,
         "notes": "Smelled like sewage near the outfall pipe, so I rated it low."},
        {"water_clarity": "cloudy", "health_rating": 3,
         "notes": "Very muddy and brown, but it rained heavily all night."},
        {"notes": "Used the new test strips for pH today, worked well."},
        {"trash_level": "high", "health_rating": 2,
         "notes": "Lots of plastic bags caught in branches, water itself looked ok."},
    ]
    for extra in tricky * 2:
        rec = _clean_record(next_id, rng, start)
        rec.update(extra)
        records.append(rec)
        next_id += 1
    rng.shuffle(records)
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.floor("min")
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/sample_observations.csv", index=False)
    print(f"Wrote {len(df)} records ({int(df.planted_error.sum())} with planted errors)"
          " to data/sample_observations.csv")
