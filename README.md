# TrueStream: AI-assisted quality checks for citizen stream science

**IEEE OneAquaHealth Global Hackathon 2026, Track 3: AI-Supported Assessment**

Citizen scientists collect valuable data about urban streams, but volunteers aren't experts and
mistakes slip in: wrong units, impossible values, contradictory answers, wrong locations, test
entries. Bad data can hide a polluted stream or trigger false alarms. TrueStream checks every
observation, explains each problem in plain language, and leaves the final decision to a human.

## How it works

1. **Validation rules**: fast, predictable checks for missing fields, impossible or
   Fahrenheit temperatures, pH missing a decimal point, "dry" streams with water measurements,
   future dates, swapped or distant GPS points, duplicates, and odor/rating conflicts.
2. **AI review (explainable)**: Claude reads each observation as a whole, including free-text
   notes, and catches what rules can't (e.g. notes describe sewage and dead fish, but the
   volunteer rated the stream 5/5). Every flag includes a plain-English explanation and a
   friendly question to ask the volunteer.
3. **Human-in-the-loop**: a reviewer confirms or dismisses every flag. The AI never changes
   data. All decisions are logged and exportable, along with the trusted dataset.
4. **Accuracy test**: we plant known errors in the data (like seeding defects in software
   testing), plus tricky-but-valid records, and measure what was caught and how many false
   alarms were raised, per layer and per error type.

## Run it locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                          # unit tests for the validation rules
export ANTHROPIC_API_KEY=sk-...    # Windows: set ANTHROPIC_API_KEY=sk-...
streamlit run app.py
```

Without an API key, choose **Offline demo** in the sidebar. That mode is a keyword heuristic for
development only, not real AI.

## Project structure

```
app.py                      Streamlit dashboard (review queue, map, report, accuracy)
truestream/generate_data.py Demo data with planted errors and tricky-but-valid records
truestream/rules.py         Layer 1: validation rules
truestream/ai_review.py     Layer 2: Claude review with explanations (cached)
truestream/evaluate.py      Accuracy test against planted errors
tests/test_rules.py         Unit tests
```

## Data

The demo uses generated records shaped like citizen stream assessments, with known errors
planted on purpose so accuracy can be measured. Uploaded CSVs with the same columns (or real
OneAquaHealth app exports mapped to them) work the same way.

## Responsible AI

AI supports, never replaces, human judgment: it only flags and explains, a person decides, every
decision is logged, and AI accuracy is measured rather than assumed.
