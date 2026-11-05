"""Run from the project root: python -m experiments.run_eval --help."""
import argparse
import json
import math
from pathlib import Path

from app.core.config import BASE_DIR, Settings
from app.services.benchmark import run_benchmark, validate_cases
from app.services.report_store import ReportWriter


def save_report(report, output_dir):
    return ReportWriter(output_dir).finish(report)


def main():
    parser = argparse.ArgumentParser(description='Generate and judge dataset cases with checkpoints. Retries may add provider calls.')
    parser.add_argument('--dataset', type=Path, default=BASE_DIR / 'experiments/sample_cases.json')
    parser.add_argument('--models', nargs='+', help='Generation model IDs; defaults to GEMINI_MODEL.')
    parser.add_argument('--output-dir', type=Path, default=BASE_DIR / 'results')
    parser.add_argument('--limit', type=int, help='Run only the first N cases.')
    parser.add_argument('--resume', type=Path, help='Reuse successful rows from a matching earlier report; retry failures only.')
    parser.add_argument('--rubric', choices=['legacy-v1', 'quality-v2'], help='Scoring rubric; defaults to EVALUATION_RUBRIC (legacy-v1 for compatibility).')
    parser.add_argument('--max-retries', type=int, default=2, help='Additional attempts per provider stage, from 0 to 3 (default 2).')
    parser.add_argument('--max-retry-delay', type=float, default=30, help='Maximum retry wait in seconds, from 0 to 60.')
    parser.add_argument('--request-interval', type=float, default=4, help='Minimum seconds between calls to the same model (0–60).')
    parser.add_argument('--model-interval', action='append', default=[], metavar='MODEL=SECONDS', help='Override the request interval for a particular model.')
    args = parser.parse_args()
    if not math.isfinite(args.request_interval) or not 0 <= args.request_interval <= 60:
        parser.error('--request-interval must be between 0 and 60')
    if not 0 <= args.max_retries <= 3:
        parser.error('--max-retries must be between 0 and 3')
    if not math.isfinite(args.max_retry_delay) or not 0 <= args.max_retry_delay <= 60:
        parser.error('--max-retry-delay must be between 0 and 60')
    intervals = {}
    for item in args.model_interval:
        try:
            name, value = item.rsplit('=', 1)
            value = float(value)
            if not name.strip() or not math.isfinite(value) or not 0 <= value <= 60:
                raise ValueError()
            intervals[name] = value
        except ValueError:
            parser.error('--model-interval must be MODEL=SECONDS with seconds between 0 and 60')
    try:
        cases = validate_cases(json.loads(args.dataset.read_text(encoding='utf-8')))
    except (OSError, ValueError):
        parser.exit(2, 'Cannot load dataset: expected a JSON array of unique id/question/reference records (1–100 cases).\n')
    if args.limit is not None:
        if args.limit < 1:
            parser.error('--limit must be at least 1')
        cases = cases[:args.limit]
    settings = Settings()
    if args.rubric:
        settings = settings.model_copy(update={'evaluation_rubric': args.rubric})
    if settings.evaluation_rubric == 'legacy-v1' and any(c.checks.model_dump(exclude_defaults=True) for c in cases):
        parser.error('This dataset has output checks; use --rubric quality-v2.')
    if not settings.has_api_key:
        parser.exit(2, 'Set GEMINI_API_KEY in the project .env file.\n')
    models = args.models or [settings.gemini_model]
    if len(set(models)) != len(models) or any(not m.strip() for m in models):
        parser.error('Model IDs must be unique and nonempty.')
    if set(intervals) - set(models) - {settings.gemini_judge_model}:
        parser.error('--model-interval must name a selected generation model or the judge model')
    prior = None
    if args.resume:
        try:
            prior = json.loads(args.resume.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            parser.exit(2, 'Cannot read the resume report.\n')
    completed = 0
    def progress(row):
        nonlocal completed
        completed += 1
        print(f"[{completed}/{len(cases) * len(models)}] {row['model']} / {row['case_id']}: {row['status']}" + (' (reused)' if row.get('reused') else ''), flush=True)
    writer = None
    try:
        writer = ReportWriter(args.output_dir)
        print('Progress file:', writer.json_path, flush=True)
        report = run_benchmark(cases, models, settings, on_result=progress, request_interval=args.request_interval, model_intervals=intervals, prior_report=prior, on_checkpoint=writer.checkpoint, max_retries=args.max_retries, max_retry_delay=args.max_retry_delay)
    except ValueError:
        parser.exit(2, 'Resume report does not match the dataset, judge, or generation settings, or scoring rubric, or has invalid completed rows.\n')
    except KeyboardInterrupt:
        if writer is not None and writer.json_path.exists():
            print('Stopped. Resume from:', writer.json_path, flush=True)
        else:
            print('Stopped before a progress file could be saved.', flush=True)
        return 130
    except OSError:
        print('Unable to save progress; run stopped. Check the output folder and any existing checkpoint.', flush=True)
        return 1
    try:
        paths = writer.finish(report)
    except KeyboardInterrupt:
        print('Export stopped. The completed JSON checkpoint remains available:', writer.json_path, flush=True)
        return 130
    except OSError:
        print('Unable to finish exporting reports; the latest valid JSON checkpoint remains available.', flush=True)
        return 1
    for summary in report['summary']:
        print(json.dumps(summary))
    print('Saved:', *paths, sep='\n')
    return 1 if any(r['status'] != 'ok' for r in report['results']) else 0


if __name__ == '__main__':
    raise SystemExit(main())
