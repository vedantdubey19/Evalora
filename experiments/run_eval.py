"""Run from the project root: python -m experiments.run_eval --help."""
import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import BASE_DIR, Settings
from app.services.benchmark import run_benchmark, validate_cases


def save_report(report, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = datetime.now(timezone.utc).strftime('eval-%Y%m%dT%H%M%S-%fZ')
    json_path = output_dir / f'{stem}.json'
    csv_path = output_dir / f'{stem}.csv'
    # Exclusive creation avoids overwriting an existing experiment.
    with json_path.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    with csv_path.open('x', newline='', encoding='utf-8') as handle:
        fields = ['case_id', 'category', 'model', 'judge_model', 'status', 'error_reason', 'generation_ms',
                  'evaluation_ms', 'total_tokens', 'correctness', 'relevance', 'clarity', 'overall']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report['results']:
            flat = {key: row.get(key) for key in fields}
            flat['total_tokens'] = (row['usage'] or {}).get('total_tokens')
            for key in ['correctness', 'relevance', 'clarity', 'overall']:
                flat[key] = (row['evaluation'] or {}).get(key)
            writer.writerow(flat)
    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser(description='Generate and judge each dataset case. Makes up to two Gemini calls per case per model.')
    parser.add_argument('--dataset', type=Path, default=BASE_DIR / 'experiments/sample_cases.json')
    parser.add_argument('--models', nargs='+', help='Generation model IDs; defaults to GEMINI_MODEL.')
    parser.add_argument('--output-dir', type=Path, default=BASE_DIR / 'results')
    parser.add_argument('--limit', type=int, help='Run only the first N cases.')
    parser.add_argument('--resume', type=Path, help='Reuse successful rows from a matching earlier report; retry failures only.')
    parser.add_argument('--request-interval', type=float, default=4, help='Minimum seconds between calls to the same model (0–60).')
    parser.add_argument('--model-interval', action='append', default=[], metavar='MODEL=SECONDS', help='Override the request interval for a particular model.')
    args = parser.parse_args()
    if not math.isfinite(args.request_interval) or not 0 <= args.request_interval <= 60:
        parser.error('--request-interval must be between 0 and 60')
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
    try:
        report = run_benchmark(cases, models, settings, on_result=progress, request_interval=args.request_interval, model_intervals=intervals, prior_report=prior)
    except ValueError:
        parser.exit(2, 'Resume report does not match the dataset, judge, or generation settings, or has invalid completed rows.\n')
    paths = save_report(report, args.output_dir)
    for summary in report['summary']:
        print(json.dumps(summary))
    print('Saved:', *paths, sep='\n')
    return 1 if any(r['status'] != 'ok' for r in report['results']) else 0


if __name__ == '__main__':
    raise SystemExit(main())
