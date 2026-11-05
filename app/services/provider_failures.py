"""Extract safe failure categories; never retain provider messages or payloads."""
import math
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx


def _delay(value):
    match = re.fullmatch(r'(\d+(?:\.\d+)?)s?', str(value))
    if not match:
        return None
    seconds = float(match.group(1))
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def classify_failure(error, *, now=None):
    code = getattr(error, 'code', None)
    retry_after = None
    daily = False
    payload = getattr(error, 'details', None)
    if isinstance(payload, dict):
        payload = payload.get('error', payload)
        details = payload.get('details', []) if isinstance(payload, dict) else []
        if isinstance(details, list):
            for detail in details:
                if not isinstance(detail, dict):
                    continue
                delay = _delay(detail.get('retryDelay', ''))
                if delay is not None:
                    retry_after = max(retry_after or 0, delay)
                violations = detail.get('violations', [])
                if isinstance(violations, list):
                    for violation in violations:
                        if isinstance(violation, dict):
                            quota = str(violation.get('quotaId', '')).lower()
                            metric = str(violation.get('quotaMetric', '')).lower()
                            daily |= 'perday' in quota or 'per_day' in quota or 'per_day' in metric
    response = getattr(error, 'response', None)
    headers = getattr(response, 'headers', None)
    if isinstance(headers, Mapping):
        value = headers.get('retry-after', '')
        delay = _delay(value)
        if delay is None and value:
            try:
                deadline = parsedate_to_datetime(str(value))
                if deadline.tzinfo is None:
                    deadline = deadline.replace(tzinfo=timezone.utc)
                delay = max(0, (deadline - (now or datetime.now(timezone.utc))).total_seconds())
            except (ValueError, TypeError, OverflowError):
                pass
        if delay is not None:
            retry_after = max(retry_after or 0, delay)
    if code == 429:
        return ('daily_quota' if daily else 'rate_limited'), retry_after
    if code == 404:
        return 'model_unavailable', None
    if code == 503:
        return 'provider_unavailable', retry_after
    if code in (500, 502, 504):
        return 'server_error', retry_after
    if code == 408 or isinstance(error, httpx.TimeoutException):
        return 'timeout', retry_after
    if isinstance(error, httpx.TransportError):
        return 'network_error', retry_after
    return 'provider_error', None
