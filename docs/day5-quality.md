# Day 5 — Evaluation quality and richer scoring

This milestone implements the selected quality track. Dashboard run controls were not added. The older live model-comparison backlog was left aside, and no Gemini calls were made.

## Versioned rubric

`legacy-v1` keeps correctness, relevance and clarity with their original arithmetic mean. Existing settings default to this rubric, and old reports without a version are interpreted as legacy-v1.

`quality-v2` adds instruction following and completeness, with integer scores from 1 to 5:

| Dimension | Weight | Meaning |
| --- | ---: | --- |
| Correctness | 35% | Accuracy relative to the question and supplied evidence |
| Instruction following | 25% | Requested format, scope, and explicit constraints |
| Relevance | 15% | Directly addresses the task |
| Completeness | 15% | Covers essential requested information without padding |
| Clarity | 10% | Understandable, concise presentation |

These weights express project policy, not an empirically calibrated ranking system. The aggregate is computed locally from validated ratings; the judge cannot submit its own overall value. Quality-v2 rejects boolean, fractional, string, and out-of-range ratings. Reasoning and a bounded limitations list accompany the scores.

The judge receives scoring anchors, instructions against rewarding verbosity/confident style, and a clear boundary between system guidance and untrusted question/answer/reference data. These prompts reduce ambiguity but do not guarantee resistance to prompt injection or accurate judgments.

## Deterministic output checks

- `expected_exact`: case-sensitive text; trim outer whitespace by default, or explicitly preserve it.
- `max_words`: count whitespace-separated words against a requested bound.
- `require_json`: accept JSON syntax only, rejecting code fences, duplicate keys and nonstandard non-finite constants.
- `json_keys`: require exactly the supplied top-level object keys, with case-sensitive names.

Checks are independent from judge ratings. An answer can receive a high judge score while failing a requested format, and a format-valid answer can be factually wrong. No supplied checks produces a null aggregate result rather than an automatic pass. No generated code, regular expression, or model-supplied assertion is executed.

`POST /check` needs no API key. `POST /evaluate/quality` combines quality-v2 judging with those checks and requires provider access. Both new routes preserve the answer's outer whitespace. The original `/evaluate` route keeps its default legacy behavior.

## Provenance and resumption

New reports record the rubric version and quality result metadata. Scores from different rubrics are not silently pooled in the dashboard: the aggregate mean and common score chart are hidden when versions differ. Detailed dimensions and check outcomes remain inspectable.

Output requirements participate in dataset fingerprints. Empty requirements are excluded from hashing to preserve the identity of older cases. Resumption requires the same rubric and verifies the overall score, scoring weights and recomputed deterministic results. Legacy mode rejects datasets with output checks so requirements cannot be silently ignored.

## Reproduce

From the repository root after installing dependencies and activating your environment:

```sh
python -m pytest -q
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
# With available Gemini API access, explicitly opt in:
python -m experiments.run_eval --dataset experiments/quality_cases.json --rubric quality-v2
```

The four-case dataset has eight provider stages before retry attempts. It was not sent to Gemini in this milestone. Eight labeled examples in `experiments/quality_examples.json` provide offline pass/fail expectations for output requirements, including correct facts in the wrong format, JSON fences, duplicate keys, and strict whitespace.

## Validation and limits

122 offline tests cover the existing project plus weighted aggregation, strict rating validation, safe malformed-judge handling, strict JSON checks, no-key checking, answer whitespace preservation, legacy fingerprints, rubric/check compatibility during resumption, system/data prompt separation, and mixed-rubric dashboard behavior.

The eight labeled output examples match their expected results. They validate deterministic mechanics, not a calibrated LLM judge. There was no human-rated model comparison, independent fact-checking, agreement study, or live provider validation. References may be wrong, models may be biased or vulnerable to injected text, and a weighted mean can hide a severe weakness in one dimension. Inspect individual ratings and checks rather than treating an aggregate as an endorsement.

CodeRabbit raised one minor CLI error-message issue. It was corrected with a regression test proving rejection before provider calls or report creation.
