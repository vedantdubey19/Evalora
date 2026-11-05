"""Bounded owner report access; filenames cannot escape the report directory."""
from pathlib import Path
import re

REPORT_NAME = re.compile(r'eval-[A-Za-z0-9-]+\.(json|csv)\Z')
MAX_REPORT_BYTES = 10_000_000


class ReportRepository:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()

    def list_reports(self):
        if not self.directory.exists():
            return []
        return [p.name for p in sorted(self.directory.glob('eval-*'), reverse=True)
                if REPORT_NAME.fullmatch(p.name) and not p.is_symlink() and p.is_file()][:200]

    def read(self, name):
        if not REPORT_NAME.fullmatch(name):
            raise ValueError('Invalid report name.')
        path = self.directory / name
        if path.is_symlink() or path.resolve().parent != self.directory:
            raise ValueError('Invalid report path.')
        with path.open('rb') as handle:
            data = handle.read(MAX_REPORT_BYTES + 1)
        if len(data) > MAX_REPORT_BYTES:
            raise ValueError('Report exceeds download limit.')
        return data, 'application/json' if path.suffix == '.json' else 'text/csv'
