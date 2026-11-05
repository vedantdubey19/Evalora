"""Measure independent, labeled cache lookups without provider calls."""
from datetime import datetime, timezone
import hashlib
import json
import math

from pydantic import BaseModel, ConfigDict, Field, StrictBool

from app.services.benchmark import Text
from app.services.cache import SemanticCache
from app.services.cache_policy import allows_semantic_reuse


class CachePair(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str = Field(min_length=1, max_length=100)
    category: str = Field(min_length=1, max_length=100)
    source: Text
    query: Text
    expected_hit: StrictBool


def validate_pairs(data):
    if not isinstance(data, list) or not 1 <= len(data) <= 1000:
        raise ValueError('Expected 1–1000 labeled pairs.')
    pairs = [CachePair.model_validate(item) for item in data]
    if len({p.id for p in pairs}) != len(pairs):
        raise ValueError('Pair IDs must be unique.')
    return pairs


def run_cache_benchmark(pairs, encoder, embedding_model, thresholds, policies=('cosine',)):
    if not thresholds or len(set(thresholds)) != len(thresholds) or any(not math.isfinite(t) or not 0 <= t <= 1 for t in thresholds):
        raise ValueError('Thresholds must be unique finite values from 0 to 1.')
    if not policies or len(set(policies)) != len(policies) or set(policies) - {'cosine', 'guarded', 'exact'}:
        raise ValueError('Choose unique supported cache policies.')
    vectors = {}
    def encode(text):
        if text not in vectors:
            vectors[text] = encoder(text)
        return vectors[text]
    results, summary = [], []
    for policy in policies:
        for threshold in thresholds:
            tp = fp = tn = fn = errors = 0
            for pair in pairs:
                row = {**pair.model_dump(), 'threshold': threshold, 'policy': policy, 'decision': None, 'actual_hit': None,
                       'similarity': None, 'status': 'embedding_error'}
                try:
                    # A fresh cache prevents unrelated pairs influencing this lookup.
                    cache = SemanticCache(encode, threshold=threshold, match_mode=policy)
                    cache.put(pair.source, 'benchmark', 'cached answer')
                    vetoed = policy == 'guarded' and pair.source != pair.query and not allows_semantic_reuse(pair.source, pair.query)
                    if policy != 'exact' and not vetoed:
                        vector = cache._vector(pair.query)
                        row['similarity'] = max(-1.0, min(1.0, float(cache.entries[0].vector @ vector)))
                    hit = cache.get(pair.query, 'benchmark') is not None
                    if pair.source == pair.query:
                        decision = 'exact_hit'
                    elif policy == 'exact':
                        decision = 'exact_miss'
                    elif vetoed:
                        decision = 'guard_veto'
                    else:
                        decision = 'semantic_hit' if hit else 'below_threshold'
                    row.update(actual_hit=hit, status='ok', decision=decision)
                    if pair.expected_hit and hit:
                        tp += 1
                    elif hit:
                        fp += 1
                    elif pair.expected_hit:
                        fn += 1
                    else:
                        tn += 1
                except Exception:
                    errors += 1
                results.append(row)
            evaluated = tp + fp + tn + fn
            summary.append({'threshold': threshold, 'policy': policy, 'pairs': len(pairs), 'evaluated': evaluated,
                            'errors': errors, 'true_positives': tp, 'false_positives': fp,
                            'true_negatives': tn, 'false_negatives': fn,
                            'accuracy': (tp + tn) / evaluated if evaluated else None,
                            'precision': tp / (tp + fp) if tp + fp else None,
                            'recall': tp / (tp + fn) if tp + fn else None,
                            'false_positive_rate': fp / (fp + tn) if fp + tn else None})
    return {'created_at': datetime.now(timezone.utc).isoformat(), 'embedding_model': embedding_model,
            'policies': list(policies), 'policy_version': 'day3-v2',
            'dataset_sha256': hashlib.sha256(json.dumps([p.model_dump() for p in pairs], sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            'scope': 'independent labeled question pairs; no answer quality measurement',
            'summary': summary, 'results': results}
