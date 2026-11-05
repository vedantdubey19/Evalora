# Day 3 — Cache safety results

## Outcome

Day 3 cache-safety implementation and measurement are complete. Caching remains opt-in per request. Once enabled, it now defaults to exact matching; guarded semantic matching requires explicit configuration. No Gemini API calls were needed for Day 3.

The original cosine baseline produced false hits for changed negation, sort order, conversion direction and comparison operators. Exact mode bypasses embeddings entirely. Guarded mode applies an ordered, case-sensitive token signature before cosine matching. It is a conservative English-oriented heuristic, not proof that two questions have the same meaning.

## Results at threshold 0.95

| Dataset | Policy | Reusable pairs found / 12 | False hits / 12 | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: |
| Original Day 2 pairs | cosine | 2 | 4 | 33.3% | 16.7% |
| Original Day 2 pairs | guarded | 2 | 0 | 100.0% | 16.7% |
| Original Day 2 pairs | exact | 1 | 0 | 100.0% | 8.3% |
| Separate challenge pairs | cosine | 7 | 8 | 46.7% | 58.3% |
| Separate challenge pairs | guarded | 3 | 0 | 100.0% | 25.0% |
| Separate challenge pairs | exact | 2 | 0 | 100.0% | 16.7% |

All **720 final-policy lookups** completed without embedding errors: 24 pairs × 5 thresholds × 3 policies × 2 datasets. Each dataset contains 12 reusable and 12 different-answer pairs. Exact matching does not use the threshold or embeddings; repeated threshold rows are retained only for a consistent report format.

At 0.95, guarded matching removed all observed false hits on both datasets. On the original pairs it found 2 reusable pairs, compared with 1 in exact mode. On the challenge set it found 3, compared with 2 in exact mode. This is a limited reuse gain, not a general paraphrase solution.

The original cosine baseline still produced 2 false hits on the challenge set even at threshold 1.0. Raising the numeric threshold alone does not preserve distinctions discarded by the embedding model, such as letter case in quoted text or identifiers.

## What changed

- `CACHE_MATCH_MODE=exact` is the default when caching is enabled. Existing `.env` credentials were not changed.
- `guarded` is an explicit optional mode. All words, Unicode symbols, punctuation, quoted text, letter case and word order are retained. Question types and modal verbs remain distinct. The only wording rewrite is lowercase “capital city” to “capital”; nonexact multiline requests are vetoed to protect code indentation. Cosine similarity must still meet the configured threshold.
- Runtime settings reject unguarded cosine mode. It remains available in offline policy comparisons.
- Exact caching works when embeddings are absent or fail. Model separation, TTL, FIFO bounds, error fallback, and opt-in request behavior remain covered by tests.
- Candidate compatibility is checked before nearest-neighbor selection, preventing a higher-scoring incompatible entry from hiding a compatible one.
- Policy names and decision labels are saved with cache reports, and the dashboard supports both new and historical reports.

## Reproduce

Run from the repository root after installing dependencies using the README setup instructions. Activate your chosen environment first.

```sh
python -m pytest -q
python -m experiments.run_cache_eval --dataset experiments/cache_pairs.json --policies cosine guarded exact
python -m experiments.run_cache_eval --dataset experiments/cache_challenge_pairs.json --policies cosine guarded exact
python -m streamlit run dashboard/app.py --server.address 127.0.0.1
```

Select **Cache accuracy** in the dashboard and choose the run and matching policy. Exact mode is selected by default when the report contains it. The first semantic run may retrieve public embedding model metadata or weights; Gemini credentials and paid calls are not needed for these experiments.

## Reports

- Original Day 2 pairs: `results/cache-20261105T160524-841975Z.json` (plus matching CSV). Dataset SHA-256: `34454e4c87ca5b606f588ac18797de0ab4a1145a6ef12b5512726ef84c9b4fb5`.
- Separate challenge pairs: `results/cache-20261105T160529-017274Z.json` (plus matching CSV). Dataset SHA-256: `0bdc6fb8b94c291e90eb0b2e141a33e8e42045585b34ff2c35413c0ab1c6e979`.

Raw reports remain ignored by Git. The public datasets and this numeric summary are tracked. The final guard is `day3-v2`. Code-review regressions led to stronger token preservation, then both datasets were rerun. The 0.95 threshold was not retuned. Labels and questions were authored within this project; treat both datasets as development sets rather than independently blinded validation.

## Limits

Zero false hits in 24 negative pairs is not a production-safety guarantee. Guarded mode can discard valid paraphrases, and the lexical rewrite/tokenizer can miss untested semantic distinctions. These tests do not measure a large corpus, multilingual prompts, context changes, multi-user isolation, stale facts, output quality, latency or cost. Exact matching avoids semantic substitutions but does not guarantee an answer is fresh or appropriate in another context.

Keep cache reuse disabled for sensitive, dynamic, context-dependent or precision-critical tasks. Guarded mode remains opt-in. The separate Day 2 model-comparison backlog is not resolved by this offline milestone.

## Verification

71 offline tests cover the earlier API and experiments plus meaning-change regressions, identical-embedding attacks, candidate filtering, exact-mode independence from embeddings, policy-aware reports, and historical dashboard compatibility. The first CodeRabbit review raised 2 major issues (meaning-bearing symbols and question intent) and 2 minor issues (benchmark veto behavior and portable reproduction instructions). All four were addressed with regression coverage. The follow-up review raised three comments citing the pre-change main revision `ca5f2f9`; they were verified as not applicable to this branch. Direct constructor/settings checks, CLI help, focused API tests, and the actual 720 successful lookups confirm the implemented contracts. The full suite passes 71 tests.
