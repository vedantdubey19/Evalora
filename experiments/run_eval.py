"""Run from the project root: python -m experiments.run_eval --help."""
import argparse
import csv
import json
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
        fields = ['case_id', 'model', 'judge_model', 'status', 'generation_ms',
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
    args = parser.parse_args()
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
    report = run_benchmark(cases, models, settings)
    paths = save_report(report, args.output_dir)
    for summary in report['summary']:
        print(json.dumps(summary))
    print('Saved:', *paths, sep='\n')
    return 1 if any(r['status'] != 'ok' for r in report['results']) else 0


if __name__ == '__main__':
    raise SystemExit(main())
