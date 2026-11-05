# Evalora local release

Evalora is complete as a single-user local evaluation application. It supports Gemini generation, legacy and five-dimension judging, deterministic answer checks, optional exact or guarded caching, batch comparison, checkpoints and resume, and a dashboard for launching and inspecting runs.

## Start and verify

From the repository root, activate your Python environment. On the development Mac use `conda activate evalora-fixed`. For a new installation follow the README and create `.env` from `.env.example` without overwriting an existing file.

```sh
make verify
make api
```

In a second terminal with the same environment:

```sh
make dashboard
```

API: <http://127.0.0.1:8000/docs>. Dashboard: <http://127.0.0.1:8501>.

`make verify` runs `pip check` and the offline suite. The integrated workflow test generates and caches an answer through FastAPI, evaluates it using quality-v2, runs the background evaluation worker, exports JSON/CSV, loads the report through the dashboard loader, and resumes it without repeating successful provider stages. Its provider is synthetic; its scores are not measurements of Gemini quality. Tests use temporary files and require no API calls or embedding downloads.

GitHub Actions installs the declared requirements on Python 3.11 and runs the same verification command. A locally passing run does not establish the remote CI outcome; inspect the repository Actions page after publication.

## Acceptance checklist

- Validated requests and sanitized provider failures for generation and judging.
- Configuration loaded locally; keys excluded from reports and Git.
- Versioned scores, weighted overall quality, and deterministic exact/JSON/word checks.
- Default exact caching; opt-in conservative semantic reuse with expiry and capacity limits.
- Retry budgets, per-model pacing, quota safeguards, atomic checkpoints, cancellation and resume.
- Dashboard input preview, request budget, explicit Start, cancellation, downloads, and result comparison.
- Offline coverage for API, cache safety, scoring, interrupted runs, report compatibility, dashboard controls, and the integrated workflow.

## External validation and scope

The separate Day 2 live comparison remains incomplete because of provider quota, model availability, and service failures. It has deliberately been left aside. Prior live smoke checks do not establish a complete model ranking or calibration of the newer quality rubric. See the milestone documents for measured results and limitations.

This release targets localhost and one user. Shared hosting requires a separate deployment scope including authentication, access controls, quotas and persistent storage. Caches are process-local and reset on restart; a dashboard server restart stops background work, while saved checkpoints remain resumable. Judge ratings require human calibration before consequential use. Reports can contain prompts and answers, so review them before sharing.

## Troubleshooting

- Missing key: edit the local `.env` and restart the application. Never put keys in commands or reports.
- `502` provider error: inspect safe reason labels in a benchmark report; wait for provider recovery or select an accessible model. Health reports configuration presence, not live API access.
- An interrupted run: resume its saved JSON with the same dataset, judge and rubric. The dashboard offers a resume view.
- No saved results: run an evaluation explicitly. Opening the dashboard never starts paid calls.
- Dependency conflict: activate the intended environment and run `python -m pip check`. Preserve the working Mac ML environment instead of mixing Conda and pip builds.
