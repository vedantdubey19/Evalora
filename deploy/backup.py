"""Create a consistent, private backup of report files and SQLite limit state."""
import argparse
import io
import os
from pathlib import Path
import sqlite3
import tarfile
import tempfile

from app.core.config import Settings


def backup(settings, destination):
    # Reports are atomically replaced; open readers see a complete checkpoint.
    descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, 'wb') as output, tarfile.open(fileobj=output, mode='w:gz') as archive:
            if settings.reports_dir.exists():
                for path in sorted(settings.reports_dir.iterdir()):
                    if path.is_file() and not path.is_symlink() and path.suffix in {'.json', '.csv'}:
                        data = path.read_bytes()
                        info = tarfile.TarInfo('reports/'+path.name)
                        info.size, info.mode = len(data), 0o600
                        archive.addfile(info, io.BytesIO(data))
            source = settings.state_dir / 'limits.sqlite3'
            if source.exists():
                with tempfile.TemporaryDirectory() as temporary:
                    target = Path(temporary) / 'limits.sqlite3'
                    with sqlite3.connect(source) as db, sqlite3.connect(target) as saved:
                        db.backup(saved)
                    archive.add(target, arcname='state/limits.sqlite3', recursive=False)
    except BaseException:
        Path(destination).unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    backup(Settings(), args.destination)
    print('Backup created. It contains private reports; store it securely.')


if __name__ == '__main__':
    main()
