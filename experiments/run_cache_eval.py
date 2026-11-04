"""Offline cache quality sweep; first use may download embedding weights."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from app.core.config import BASE_DIR, Settings
from app.services.cache import SentenceEncoder
from app.services.cache_benchmark import run_cache_benchmark, validate_pairs


def save_cache_report(report, directory):
    directory.mkdir(parents=True, exist_ok=True)
    stem = datetime.now(timezone.utc).strftime('cache-%Y%m%dT%H%M%S-%fZ')
    json_path, csv_path = directory / f'{stem}.json', directory / f'{stem}.csv'
    with json_path.open('x', encoding='utf-8') as file:
        json.dump(report, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write('\n')
    with csv_path.open('x', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(report['results'][0]))
        writer.writeheader()
        writer.writerows(report['results'])
    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser(description='Measure cache matching on labeled pairs without LLM API calls.')
    parser.add_argument('--dataset', type=Path, default=BASE_DIR / 'experiments/cache_pairs.json')
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.85, 0.90, 0.95, 0.98, 1.0])
    parser.add_argument('--output-dir', type=Path, default=BASE_DIR / 'results')
    args = parser.parse_args()
    try:
        pairs = validate_pairs(json.loads(args.dataset.read_text(encoding='utf-8')))
        settings = Settings()
        report = run_cache_benchmark(pairs, SentenceEncoder(settings.embedding_model), settings.embedding_model, args.thresholds)
    except (OSError, ValueError):
        parser.exit(2, 'Invalid pair dataset or threshold settings.\n')
    paths = save_cache_report(report, args.output_dir)
    for row in report['summary']:
        print(json.dumps(row))
    print('Saved:', *paths, sep='\n')
    return 1 if any(r['status'] != 'ok' for r in report['results']) else 0


if __name__ == '__main__':
    raise SystemExit(main())
