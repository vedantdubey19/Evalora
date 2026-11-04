# Day 2 results — 5 November 2026

## Status

Day 2 tooling and the offline cache experiment are complete. The live two-model evaluation remains incomplete because of provider quota and availability failures. No model ranking is justified by these partial results.

## What was added

- A 24-case public evaluation dataset: six cases each for reasoning, factual answers, coding, and instruction following.
- A 24-pair labeled cache dataset: 12 reusable pairs and 12 different-answer pairs.
- Five cache thresholds, with confusion matrices, precision, recall, false-positive rates, and per-pair outcomes.
- Dataset fingerprints, temperature-zero benchmark generation, per-model request pacing, safe error labels, and validated resumption of successful cases.
- Dashboard category filters and cache accuracy inspection.

## Live model comparison

Fixed judge: `gemini-3.1-flash-lite`. Dataset SHA-256: `2a19313009759e2c6291d5e0c85bdf7bb4c78fd4f7783a315234ccb6b360ef23`. Generation temperature: 0. Caching disabled. This is a single run, not a statistical benchmark.

| Generation model | Dataset cases | Completed evaluations | Failures |
| --- | ---: | ---: | ---: |
| gemini-2.5-flash | 24 | 4 | 20 |
| gemini-3.1-flash-lite | 24 | 1 | 23 |

Gemini 2.5 Flash returned HTTP 429 with the explicitly identified daily free-tier quota `GenerateRequestsPerDayPerProjectPerModel-FreeTier`, value 20. A diagnostic check of Gemini 3.1 Flash-Lite returned HTTP 503. Earlier probing also showed Gemini 2.5 Flash-Lite returning 404 despite being listed. A model listing alone does not prove it can currently serve generation requests.

The saved report retains successful answers and marks failed rows separately. The original attempt preceded the final 503 circuit-breaker improvement, so many outage rows have the generic safe label `provider_error`. New runs stop further calls to a model on 429, 404, or 503; a judge failure with one of those codes also stops further generation for that run.

Successful rows scored 5/5 in this partial sample, but the low completion counts, different successful subsets, and missing cases make that unsuitable for comparing quality. One candidate also served as judge, which can introduce bias. Latency and token metrics in the raw report are diagnostics for completed generations, not reliable model rankings or monetary costs.

Raw reports remain local under `results/` and are ignored by Git:

- `eval-20261104T202615-746877Z.json`
- `eval-20261104T202615-746877Z.csv`

## Cache matching results

Embedding model: `sentence-transformers/all-MiniLM-L6-v2`. Dataset SHA-256: `34454e4c87ca5b606f588ac18797de0ab4a1145a6ef12b5512726ef84c9b4fb5`. All 120 lookups completed without embedding errors.

| Threshold | Correct hits | False hits | Missed reusable pairs | Precision | Recall | False-positive rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.85 | 9 | 7 | 3 | 56.2% | 75.0% | 58.3% |
| 0.90 | 6 | 6 | 6 | 50.0% | 50.0% | 50.0% |
| 0.95 | 2 | 4 | 10 | 33.3% | 16.7% | 33.3% |
| 0.98 | 1 | 3 | 11 | 25.0% | 8.3% | 25.0% |
| 1.00 | 1 | 0 | 11 | 100.0% | 8.3% | 0.0% |

At the current 0.95 threshold, 4 of 12 different-answer pairs were incorrectly reused, while only 2 of 12 reusable pairs were found. False-hit cases: `negative-negation`, `negative-sort`, `negative-direction`, `negative-operator`.

A high cosine score alone did not reliably distinguish changes in meaning. Keep caching opt-in and disabled for precision-critical questions. The default threshold was not changed based on this small development dataset. Threshold 1.0 produced no false hits here but found only one reusable pair; that does not establish production safety or prove it always behaves like exact matching.

These tests classify human-labeled question pairs; they do not grade cached answer quality. Each pair uses a fresh cache with one source question. Large-corpus nearest-neighbor behavior, repeated queries, embedding cold-start latency, and production traffic were not measured. Use separate held-out pairs before tuning any policy.

Raw cache reports: `results/cache-20261104T201637-221645Z.json` and `.csv`.

## Resume the live comparison

After provider access returns, run from the project root:

```sh
GEMINI_JUDGE_MODEL=gemini-3.1-flash-lite python -m experiments.run_eval \
  --dataset experiments/day2_cases.json \
  --models gemini-2.5-flash gemini-3.1-flash-lite \
  --request-interval 4 --model-interval gemini-2.5-flash=13 \
  --resume results/eval-20261104T202615-746877Z.json
```

The five successful evaluations are reused, and failed cases are retried. Resume compatibility checks require the same dataset fingerprint, judge, and generation temperature, and validate each reused answer, score, and latency. The resumed report records its origin and reuse count. Account quotas can change; the request intervals above are examples, not a guarantee of capacity.

View active quotas in [Google AI Studio](https://aistudio.google.com/rate-limit). Google documents per-project limits and daily quota reset behavior in its [rate limit documentation](https://ai.google.dev/gemini-api/docs/rate-limits).

## Validation

43 offline tests cover the original API and dashboard plus dataset validation, cache confusion matrices, missing-rate handling, embedding errors, paced latency, provider circuit breaking, and compatible/incompatible resumption. The core Day 2 diff received a CodeRabbit review with 0 issues. The final review raised 2 minor issues involving malformed resumed usage and unknown pacing-override model names. Both were fixed, with regression tests confirming rejection before provider calls. The dashboard was also checked against the actual saved reports.
