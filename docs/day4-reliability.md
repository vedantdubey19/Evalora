# Day 4 — Reliable evaluation runs

Day 4 adds durable progress checkpoints, bounded transient retries, and honest failure accounting. Validation uses simulated providers; no Gemini requests, API credentials, or billing changes were needed.

## Completed behavior

- Write a complete initial JSON snapshot before constructing or calling a provider.
- Atomically replace the same JSON file after every processed case. Temporary files are hidden from the dashboard and cleaned after success or failure.
- On cancellation, retain completed cases and mark the report interrupted. Exit cleanly with code 130.
- Stop provider calls on checkpoint failure, preserving the last valid snapshot.
- Resume only validated successful cases from a matching dataset, judge, and generation configuration. Pending, skipped and interrupted cases remain eligible to retry.
- Retry known temporary 5xx, network, and timeout failures within a per-stage budget. Honor short server retry delays, including HTTP dates. Never retry a recognized daily quota or a wait longer than the configured cap.
- Open a per-model circuit after nonretryable or exhausted blocking failures. A blocked judge causes subsequent unevaluated cases to be skipped without generation calls.
- Separate attempted failures, skips, pending work, reused cases, and successful evaluations. Keep old reports readable without inventing missing invocation counts.

## Defaults and limits

The CLI defaults to two additional attempts per provider stage and a 30-second retry-delay cap. The maximum selectable budget is three additional attempts; the delay cap cannot exceed 60 seconds. Backoff includes jitter. An unknown 429 without a usable server retry delay stops the model immediately. Existing per-model pacing also remains active.

These retries apply only to the experiment runner. The API endpoint keeps its single-attempt behavior. Provider exceptions are mapped to safe reason labels; messages, URLs, credentials and raw quota payloads are not stored in reports.

The 24-case, two-model dataset has 96 generation/judge stages. Two additional attempts can increase the theoretical upper bound to 288 provider invocations, although circuits can stop the run much earlier. Use `--max-retries 0` to disable retries.

## Offline validation

The full suite has **96 passing tests**. Day 4 tests cover:

| Scenario | Verified outcome |
| --- | --- |
| Initial checkpoint fails | No provider is constructed or called |
| Normal two-case run | Every call sees the previous saved progress; final JSON matches the result |
| Cancellation after a completed case | Completed work remains in valid JSON; remaining work stays pending |
| Cancellation before the first checkpoint | No saved file is falsely claimed |
| Cancellation during CSV export | Completed JSON remains available |
| Cancellation inside a provider call | Interrupted case is marked and CLI exits 130 without a raw traceback |
| Resume an interrupted run | Successful case is reused with zero new calls; remaining cases run |
| Atomic replacement fails | Previous JSON stays unchanged and temporary file is removed |
| Disk fails after the first case | Further calls stop; the previous checkpoint remains valid |
| Two temporary failures then recovery | Three generation attempts, two retry waits, and one judging attempt |
| Temporary failures exhaust budget | One attempted case failure followed by skipped cases |
| Judge outage exhausts budget | Later cases skip generation across models |
| Known daily quota | One attempted request, no retries or retry sleeps |
| Ambiguous or long-delay limit | No premature retries; later cases are skipped |
| Short server retry delay | The required delay is respected |
| HTTP-date Retry-After | Future deadline is extracted correctly |
| In-progress dashboard | Pending and skipped cases display without errors |
| Historical dashboard reports | Previous reports still load; missing call counts are unknown |

Latency tests use a controllable clock. Provider durations include all attempted calls for a stage, but exclude pacing and retry sleeps. All failure simulations are deterministic and use no external model downloads or provider services.

## Run and resume

From the repository root, after activating your environment and installing dependencies:

```sh
python -m pytest -q
python -m experiments.run_eval --max-retries 2 --max-retry-delay 30
python -m experiments.run_eval --resume results/EXACT_CHECKPOINT_FILENAME.json
python -m streamlit run dashboard/app.py --server.address 127.0.0.1
```

Keep the original dataset, model selection and judge configuration when resuming the same comparison. Each resume writes a new report rather than overwriting its source. A final CSV is exported only when processing finishes normally; an interrupted JSON file is sufficient for resumption.

## What the counts mean

`processed_cases` includes successes, attempted failures and explicit skips. `finished` describes the runner, not model quality. Provider-attempt and retry counters cover this run, while reused answers retain their original latency and token observations. Counts describe wrapper invocations and are not guaranteed billing or wire-request totals.

Legacy reports cannot reliably distinguish skipped rows from actual failed attempts, so they retain legacy accounting and an explanatory dashboard note. New reports use `accounting_version: day4-v1` and `schema_version: 2`.

## Remaining limits

A provider request and local file write cannot be one transaction. A process killed while a request is in flight may require that case to run again; provider usage may still have occurred. Checkpoints protect completed writes against ordinary interruption, not storage hardware failure. A disk write that fails before replacement preserves only the previous successful snapshot.

A case interrupted after generation but before successful judging is saved for inspection but is not treated as a fully reusable evaluation. Daily quotas and provider availability still require external recovery; this work does not complete Day 2's remaining live evaluations or bypass provider limits.

CodeRabbit reviewed the core Day 4 diff and raised 0 issues. Final cancellation-boundary and legacy-label regression checks also pass.
