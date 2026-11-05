"""Atomic JSON checkpoints and CSV exports; reports never contain configuration."""
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import tempfile
from uuid import uuid4


def atomic_text(path, text):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.eval-', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class ReportWriter:
    def __init__(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = datetime.now(timezone.utc).strftime('eval-%Y%m%dT%H%M%S-%fZ-') + uuid4().hex
        self.json_path = output_dir / f'{stem}.json'
        self.csv_path = output_dir / f'{stem}.csv'

    def checkpoint(self, report):
        atomic_text(self.json_path, json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

    def finish(self, report):
        self.checkpoint(report)
        fields = ['case_id', 'category', 'model', 'judge_model', 'status', 'error_reason',
                  'reused', 'generation_attempts', 'evaluation_attempts', 'retries',
                  'generation_ms', 'evaluation_ms', 'total_tokens',
                  'correctness', 'relevance', 'clarity', 'overall']
        output = io.StringIO(newline='')
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in report['results']:
            flat = {key: row.get(key) for key in fields}
            flat['total_tokens'] = (row.get('usage') or {}).get('total_tokens')
            for key in ['correctness', 'relevance', 'clarity', 'overall']:
                flat[key] = (row.get('evaluation') or {}).get(key)
            writer.writerow(flat)
        atomic_text(self.csv_path, output.getvalue())
        return self.json_path, self.csv_path
