"""Unit tests for the validation rules. Run: pytest -q"""
from datetime import datetime, timedelta

import pandas as pd

from truestream.rules import (check_dry_contradiction, check_location, check_odor_rating, check_ph,
                              check_required, check_temperature, check_timestamp, run_rules)


def make(**overrides):
    base = {"observation_id": "T1", "volunteer_id": "V01", "site_id": "S01", "site_name": "x",
            "timestamp": pd.Timestamp("2026-09-20 10:00"), "latitude": 38.7605, "longitude": -9.1702,
            "water_temp_c": 15.0, "ph": 7.2, "water_clarity": "clear", "odor": "none", "flow": "slow",
            "trash_level": "low", "insects_count": 10, "health_rating": 4, "notes": ""}
    base.update(overrides)
    return pd.Series(base)


def rules_of(issues):
    return [i["rule"] for i in issues]


def test_valid_record_has_no_issues():
    df = pd.DataFrame([make()])
    assert run_rules(df, now=datetime(2026, 10, 1)).empty


def test_missing_required_field():
    assert rules_of(check_required(make(flow=None))) == ["missing_field"]


def test_fahrenheit_detected():
    assert rules_of(check_temperature(make(water_temp_c=68.0))) == ["temp_fahrenheit"]


def test_impossible_temperature():
    assert rules_of(check_temperature(make(water_temp_c=150.0))) == ["temp_impossible"]
    assert rules_of(check_temperature(make(water_temp_c=-12.0))) == ["temp_impossible"]


def test_temperature_boundaries_are_valid():
    assert check_temperature(make(water_temp_c=0.0)) == []
    assert check_temperature(make(water_temp_c=30.0)) == []


def test_ph_missing_decimal():
    issues = check_ph(make(ph=72))
    assert rules_of(issues) == ["ph_missing_decimal"]
    assert "7.2" in issues[0]["volunteer_question"]


def test_dry_but_measured():
    assert rules_of(check_dry_contradiction(make(flow="dry"))) == ["dry_but_measured"]
    assert check_dry_contradiction(make(flow="dry", water_temp_c=None, ph=None)) == []


def test_future_timestamp():
    future = make(timestamp=pd.Timestamp(datetime.now() + timedelta(days=30)))
    assert rules_of(check_timestamp(future, datetime.now())) == ["future_timestamp"]


def test_swapped_coordinates():
    assert rules_of(check_location(make(latitude=-9.1702, longitude=38.7605))) == ["gps_swapped"]


def test_far_from_site():
    assert rules_of(check_location(make(latitude=38.95))) == ["gps_far"]


def test_odor_rating_conflict():
    assert rules_of(check_odor_rating(make(odor="sewage", health_rating=5))) == ["odor_rating_conflict"]
    assert check_odor_rating(make(odor="sewage", health_rating=2)) == []


def test_duplicate_detected():
    a = make(observation_id="A")
    b = make(observation_id="B", timestamp=a.timestamp + timedelta(minutes=3))
    out = run_rules(pd.DataFrame([a, b]), now=datetime(2026, 10, 1))
    assert list(out.rule) == ["duplicate"] and list(out.observation_id) == ["B"]
