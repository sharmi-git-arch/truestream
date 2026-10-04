"""
Layer 2: AI review with explanations.

The AI reads each observation *as a whole* (fields + free-text notes + any rule findings)
and catches what rules can't, e.g. notes that contradict the recorded fields.

Modes:
  - "gemini": real AI via Google's Gemini API, free tier (needs GEMINI_API_KEY)
  - "claude": real AI via the Anthropic API (needs ANTHROPIC_API_KEY)
  - "offline": simple keyword heuristic so the app still runs without a key.
               Clearly labelled in the UI; NOT real AI. Use for development only.

Results are cached in data/ai_cache.json so you never pay twice for the same record.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd

MODEL = os.getenv("TRUESTREAM_MODEL", "claude-haiku-4-5-20251001")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_PAUSE_SEC = 5  # free tier allows only a few requests per minute
CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "ai_cache.json"
BATCH_SIZE = 10
MIN_CONFIDENCE = 0.7  # AI flags below this confidence are not shown to reviewers
ERRORS: list[str] = []  # API errors from the last run, shown in the app

FIELDS = ["observation_id", "site_name", "timestamp", "water_temp_c", "ph", "water_clarity",
          "odor", "flow", "trash_level", "insects_count", "health_rating", "notes"]

SYSTEM_PROMPT = """You review citizen-science stream assessments for data quality.
Volunteers are not experts. Your job is to SUPPORT a human reviewer, never to replace them.
Every flag costs a researcher time, so only flag records that are genuinely likely to be wrong.
Most observations are fine: typically fewer than 1 in 10 needs review.

FLAG only when there is clear evidence of a data-quality problem, such as:
- free-text notes that clearly contradict the recorded fields (e.g. notes describe sewage smell,
  foam or dead fish, but odor is "none" and health rating is 5)
- test, joke or junk entries (e.g. "asdf", "test 123", "just trying the app")
- values that cannot all be true at the same time

These are NORMAL and must NOT be flagged:
- empty or very short notes
- muddy or cloudy water, especially after rain
- trash, algae, leaves, ducks, or other everyday observations
- a low health rating with a matching smell, trash or murky water
- health ratings of 3, 4 or 5 for ordinary-looking water (ratings are subjective)
- insect counts anywhere from 0 to 40
- notes that mention testing equipment (e.g. "test strips", "pH test kit")
- problems already listed under rule_findings (rules handle those; do not repeat them)

When unsure, do not flag. Use confidence honestly: 0.9+ only for obvious problems.

Respond with ONLY a JSON array, no markdown, one object per observation:
[{"observation_id": "...",
  "flag": true|false,
  "issue_type": "short_snake_case_label or empty",
  "explanation": "1-2 plain-English sentences a non-expert understands",
  "volunteer_question": "a friendly question to ask the volunteer, or empty",
  "confidence": 0.0-1.0}]"""


def _key(rec: dict, mode: str) -> str:
    # include the prompt, so editing the prompt automatically triggers a fresh review
    raw = mode + SYSTEM_PROMPT + json.dumps(rec, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode()).hexdigest()


def _load_cache() -> dict:
    try:
        return json.loads(CACHE_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=1))


def _parse_json_array(text: str) -> list[dict]:
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start, end = text.find("["), text.rfind("]")
    return json.loads(text[start:end + 1])


def _call_claude(batch: list[dict]) -> list[dict]:
    import anthropic  # imported lazily so offline mode needs no key

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(batch, default=str, indent=1)}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    return _parse_json_array(text)


def _call_gemini(batch: list[dict]) -> list[dict]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT,
                                         response_mime_type="application/json")
    for attempt in range(4):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL, contents=json.dumps(batch, default=str, indent=1), config=config)
            return _parse_json_array(resp.text)
        except Exception as e:
            if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and attempt < 3:
                time.sleep(20 * (attempt + 1))  # free-tier rate limit: wait, then retry
                continue
            raise


# ---- offline heuristic (development only, not AI) ---------------------------------------
_BAD_WORDS = ["sewage", "rotten", "foam", "dead fish", "oily", "sheen", "toilet", "chemical smell"]
_JUNK = ["test", "asdf", "lol", "trying the app"]


def _offline_review(rec: dict) -> dict:
    notes = str(rec.get("notes") or "").lower()
    out = {"observation_id": rec["observation_id"], "flag": False, "issue_type": "",
           "explanation": "", "volunteer_question": "", "confidence": 0.5}
    if any(w in notes for w in _JUNK):
        out.update(flag=True, issue_type="junk_entry", confidence=0.7,
                   explanation="The notes look like a test or practice entry rather than a real observation.",
                   volunteer_question="Was this a practice entry? If so, we can remove it.")
    elif any(w in notes for w in _BAD_WORDS) and (rec.get("health_rating") or 0) >= 4:
        out.update(flag=True, issue_type="notes_contradict_fields", confidence=0.7,
                   explanation="The notes describe signs of pollution, but the recorded odor/clarity "
                               "and health rating say the stream is healthy.",
                   volunteer_question="Your notes mention signs of pollution. Should the health rating be lower?")
    return out
# ------------------------------------------------------------------------------------------


def review(df: pd.DataFrame, rule_issues: pd.DataFrame, mode: str = "claude",
           progress=None) -> pd.DataFrame:
    """Return a DataFrame of AI findings (one row per flagged observation)."""
    findings_by_obs = rule_issues.groupby("observation_id")["message"].apply(list).to_dict()
    records = []
    for row in df[[c for c in FIELDS if c in df.columns]].to_dict("records"):
        row["rule_findings"] = findings_by_obs.get(row["observation_id"], [])
        records.append(row)

    ERRORS.clear()
    results: list[dict] = []
    if mode == "offline":
        results = [_offline_review(r) for r in records]
    else:
        call = _call_gemini if mode == "gemini" else _call_claude
        cache = _load_cache()
        todo = [r for r in records if _key(r, mode) not in cache]
        failed: dict[str, dict] = {}
        for i in range(0, len(todo), BATCH_SIZE):
            batch = todo[i:i + BATCH_SIZE]
            if mode == "gemini" and i > 0:
                time.sleep(GEMINI_PAUSE_SEC)
            try:
                answers = {a["observation_id"]: a for a in call(batch)}
                for r in batch:  # only successful answers are cached
                    cache[_key(r, mode)] = answers.get(
                        r["observation_id"], {"observation_id": r["observation_id"], "flag": False})
                _save_cache(cache)
            except Exception as e:  # keep going; failed batches are retried on the next run
                ERRORS.append(str(e)[:300])
                for r in batch:
                    failed[_key(r, mode)] = {"observation_id": r["observation_id"], "flag": False}
            if progress:
                progress(min(1.0, (i + BATCH_SIZE) / max(len(todo), 1)))
        results = [cache.get(_key(r, mode)) or failed[_key(r, mode)] for r in records]

    rows = []
    for a in results:
        conf = a.get("confidence")
        conf = float(conf) if isinstance(conf, (int, float)) else 1.0
        if a.get("flag") and conf >= MIN_CONFIDENCE:
            rows.append({
                "observation_id": a["observation_id"],
                "layer": "ai",
                "rule": a.get("issue_type") or "ai_flag",
                "severity": "warning",
                "message": a.get("explanation", ""),
                "volunteer_question": a.get("volunteer_question", ""),
                "confidence": a.get("confidence"),
            })
    cols = ["observation_id", "layer", "rule", "severity", "message", "volunteer_question", "confidence"]
    return pd.DataFrame(rows, columns=cols)
