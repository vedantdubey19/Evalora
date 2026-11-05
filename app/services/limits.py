"""SQLite admission control shared by the API, dashboard and provider clients.

Fixed windows use UTC epoch boundaries. Failed provider attempts consume budget.
Only trusted fixed bucket names are stored; request headers and secrets are not.
"""
from contextlib import contextmanager
from pathlib import Path
import sqlite3
import time
from uuid import uuid4


class LimitExceeded(Exception):
    def __init__(self, retry_after, reason='rate_limited'):
        self.retry_after = max(1, int(retry_after))
        self.reason = reason
        super().__init__('Application request limit reached.')


class LimitStore:
    def __init__(self, directory, clock=time.time):
        self.path = Path(directory) / 'limits.sqlite3'
        self.clock = clock
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self._connection() as connection:
            connection.execute('CREATE TABLE IF NOT EXISTS counters (bucket TEXT PRIMARY KEY, window INTEGER NOT NULL, count INTEGER NOT NULL)')
            connection.execute('CREATE TABLE IF NOT EXISTS leases (id TEXT PRIMARY KEY, expires REAL NOT NULL)')
        self.path.chmod(0o600)

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        try:
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _consume(connection, bucket, maximum, seconds, now):
        window = int(now // seconds)
        row = connection.execute('SELECT window, count FROM counters WHERE bucket=?', (bucket,)).fetchone()
        count = row[1] if row and row[0] == window else 0
        if count >= maximum:
            raise LimitExceeded((window + 1) * seconds - now)
        connection.execute('INSERT INTO counters VALUES (?, ?, ?) ON CONFLICT(bucket) DO UPDATE SET window=excluded.window, count=excluded.count',
                           (bucket, window, count + 1))

    def consume(self, bucket, maximum, seconds=60):
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            try:
                self._consume(connection, bucket, maximum, seconds, self.clock())
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    @contextmanager
    def provider_slot(self, daily_limit, concurrent_limit, timeout_seconds):
        lease = uuid4().hex
        now = self.clock()
        with self._connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            try:
                connection.execute('DELETE FROM leases WHERE expires <= ?', (now,))
                if connection.execute('SELECT COUNT(*) FROM leases').fetchone()[0] >= concurrent_limit:
                    raise LimitExceeded(5)
                try:
                    self._consume(connection, 'provider-day', daily_limit, 86400, now)
                except LimitExceeded as error:
                    error.reason = 'daily_quota'
                    raise
                connection.execute('INSERT INTO leases VALUES (?, ?)', (lease, now + timeout_seconds + 30))
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        try:
            yield
        finally:
            with self._connection() as connection:
                connection.execute('DELETE FROM leases WHERE id=?', (lease,))
