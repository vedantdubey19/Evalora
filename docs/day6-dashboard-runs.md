# Day 6 — Dashboard evaluation workflow

Day 6 adds a local workflow to choose a dataset, preview validation, set model/rubric/retry options, start a background evaluation, monitor its checkpoint, cancel, resume, and download JSON/CSV results. All development checks used fake providers. No live Gemini calls were started.

## Workflow

1. Start Streamlit using the README instructions and select **Run evaluation**.
2. Choose sample questions, quality cases, or a UTF-8 JSON upload. Uploads are capped at 2 MB and 100 cases. The same case schema, unique IDs, references, and output-check validation apply as the CLI.
3. Inspect the first five rows and choose the question count, up to four generation models, judge, rubric, retries, and pacing interval. Quality output requirements cannot be silently run with the legacy rubric.
4. Review the provider-call budget and data-sharing notice. Only pressing **Start evaluation (uses Gemini)** starts work.
5. Watch the latest run update while the page is open. Download a valid JSON snapshot during a run; a CSV is available after normal completion.
6. Cancel if needed. Between calls, request pacing and retry waits are interruptible. An HTTP request already in flight cannot be forcibly terminated; the worker waits for a response or the configured timeout and then stops before another call.
7. Resume from a saved results JSON. The dataset and options are reconstructed, original pacing overrides are preserved, and saved successful scores are revalidated. A new report is written, preserving its source.

## Safety and lifecycle

The manager allows one active worker per dashboard process and prevents double starts across that process's tabs. Workers never call Streamlit or access session state. They use the existing benchmark engine and atomic report writer; UI polling only reads sanitized job snapshots.

Jobs continue when a tab closes. A server-process restart stops daemon workers and loses their in-memory job registry, but existing checkpoints remain. A report marked running can belong to an orphaned process or a still-active CLI; resumption requires acknowledgement that the original was stopped. The manager does not coordinate separate dashboard processes or CLI runs.

Uploads are parsed as data, never executed, and are not saved as separate raw files. Started reports contain their case data and must be treated as private when appropriate. Dataset validation identifies rows and fields without echoing whole invalid records. Keys are loaded from the existing `.env` configuration and are never placed in job snapshots. Unexpected worker errors use generic messages, not provider exceptions, URLs, or credentials.

The app remains localhost-only and single-user. Authentication, multiple tenants, process-wide job queues, and deployment infrastructure are outside this milestone. The maximum call count is a bound on provider-stage invocations, not a price quote or guaranteed billing/request count. Quotas, model access and provider outages remain external constraints.

## Validation

The full project suite has **140 passing offline tests**. Day 6 coverage includes:

| Check | Result |
| --- | --- |
| Merely opening the run page | No provider calls or worker launch |
| Invalid/oversized JSON and missing fields | Rejected with bounded row-level messages |
| Missing key, duplicate models, incompatible rubric/resume | Rejected before a worker starts |
| Successful fake evaluation | Valid JSON and CSV with dashboard provenance |
| Second start during active work | Blocked |
| Cancel during an in-flight fake request | No judge or subsequent provider call begins |
| Cancel during pacing or retry wait | Wait ends without another provider invocation |
| Unexpected worker exception | Safe generic error; key and raw message absent |
| Resume from saved uploaded-case data | Dataset/options reconstructed; complete cases use zero new attempts |
| Running checkpoint | Original-process acknowledgement required before Start |
| Out-of-folder checkpoint | Rejected |
| UI start/resume/download paths | Pass with fake providers |

The existing checkpoint, provider-retry, quota, rubric and cache tests continue to pass. Tests exercise concurrency using synchronization events rather than real services or long sleeps.

## Reproduce

From the repository root after installing dependencies and activating your environment:

```sh
python -m pytest -q
python -m streamlit run dashboard/app.py --server.address 127.0.0.1
```

The dashboard workflow requires a configured API key only when a real run is started. Tests inject a fake provider and a dummy key; they never alter `.env` or call Gemini.

Implementation follows Streamlit's [worker-thread guidance](https://docs.streamlit.io/develop/concepts/design/multithreading) and uses [fragments](https://docs.streamlit.io/develop/concepts/architecture/fragments) for UI polling. Background workers do not display elements or mutate UI session state.

CodeRabbit raised one minor late-cancellation race. It was fixed with regressions confirming finished, cancelled and failed jobs cannot be overwritten by a late Cancel action.
