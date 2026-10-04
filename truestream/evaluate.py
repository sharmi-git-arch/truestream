"""
Accuracy test: compare what TrueStream flagged with the errors we planted on purpose.

Like seeding defects in software testing:
  true positive  = planted error that was flagged
  false negative = planted error that was missed
  false positive = clean record that was flagged (a false alarm)
"""
from __future__ import annotations

import pandas as pd


def _score(truth: pd.Series, flagged: set[str]) -> dict:
    pred = truth.index.isin(list(flagged))
    tp = int((truth & pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"caught": tp, "missed": fn, "false_alarms": fp,
            "precision": round(precision, 3), "recall": round(recall, 3)}


def evaluate(df: pd.DataFrame, issues: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (summary by layer, breakdown by planted error type)."""
    truth = df.set_index("observation_id")["planted_error"].astype(bool)
    by_layer = {
        "Rules only": set(issues.loc[issues.layer == "rules", "observation_id"]),
        "AI only": set(issues.loc[issues.layer == "ai", "observation_id"]),
        "Rules + AI (TrueStream)": set(issues.observation_id),
    }
    summary = pd.DataFrame({name: _score(truth, ids) for name, ids in by_layer.items()}).T

    planted = df[df.planted_error.astype(bool)][["observation_id", "planted_error_type"]]
    rows = []
    for kind, grp in planted.groupby("planted_error_type"):
        ids = set(grp.observation_id)
        rows.append({
            "planted error": kind,
            "planted": len(ids),
            "caught by rules": len(ids & by_layer["Rules only"]),
            "caught by AI": len(ids & by_layer["AI only"]),
            "caught overall": len(ids & by_layer["Rules + AI (TrueStream)"]),
        })
    return summary, pd.DataFrame(rows)


def false_alarms(df: pd.DataFrame, issues: pd.DataFrame) -> pd.DataFrame:
    """Clean records that were flagged anyway, with the reason, to learn from mistakes."""
    clean_ids = set(df.loc[~df.planted_error.astype(bool), "observation_id"])
    fa = issues[issues.observation_id.isin(clean_ids)]
    cols = [c for c in ["observation_id", "layer", "rule", "message"] if c in fa]
    out = fa[cols].merge(df[["observation_id", "notes", "health_rating"]], on="observation_id")
    return out.rename(columns={"rule": "flag", "message": "reason"})
