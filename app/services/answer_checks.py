"""Deterministic output checks. Never execute generated code or expressions."""
import json
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, model_validator

Key = Annotated[StrictStr, Field(min_length=1, max_length=100)]


class AnswerChecks(BaseModel):
    model_config = ConfigDict(extra='forbid')
    expected_exact: str | None = Field(default=None, max_length=20000)
    trim_outer_whitespace: StrictBool = True
    require_json: StrictBool = False
    json_keys: list[Key] | None = Field(default=None, max_length=30)
    max_words: int | None = Field(default=None, ge=1, le=1000, strict=True)

    @model_validator(mode='after')
    def unique_keys(self):
        if self.json_keys is not None and len(set(self.json_keys)) != len(self.json_keys):
            raise ValueError('JSON keys must be unique.')
        return self


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def _nonfinite(_):
    raise ValueError('Non-finite JSON number')


def check_answer(answer: str, spec: AnswerChecks):
    checks = []
    def add(name, passed, detail):
        checks.append({'check': name, 'passed': bool(passed), 'detail': detail})
    if spec.expected_exact is not None:
        actual, expected = answer, spec.expected_exact
        if spec.trim_outer_whitespace:
            actual, expected = actual.strip(), expected.strip()
        add('exact_match', actual == expected, 'Case-sensitive exact text comparison; outer whitespace ' + ('ignored.' if spec.trim_outer_whitespace else 'preserved.'))
    if spec.max_words is not None:
        count = len(answer.split())
        add('max_words', count <= spec.max_words, f'{count} whitespace-separated words; limit {spec.max_words}.')
    if spec.require_json or spec.json_keys is not None:
        parsed, valid = None, False
        try:
            parsed = json.loads(answer, object_pairs_hook=_object, parse_constant=_nonfinite)
            valid = True
        except (ValueError, RecursionError):
            pass
        add('valid_json', valid, 'Strict JSON only: no Markdown fences, duplicate keys, or non-finite constants.')
        if spec.json_keys is not None:
            add('json_keys', valid and isinstance(parsed, dict) and set(parsed) == set(spec.json_keys),
                'Top-level object keys must exactly match the requested key set.')
    return {'checks_passed': all(c['passed'] for c in checks) if checks else None,
            'deterministic_checks': checks}
