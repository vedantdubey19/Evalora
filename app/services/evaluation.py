import json

from app.services.answer_checks import AnswerChecks, check_answer
from app.services.llm_client import ProviderError
from app.services.rubrics import Scores, WEIGHTS, overall_score, score_schema


def evaluate(client, question, answer, reference, *, rubric=None, checks=None):
    rubric = rubric or client.settings.evaluation_rubric
    schema = score_schema(rubric)
    spec = checks or AnswerChecks()
    payload = {'question': question, 'answer': answer, 'reference': reference}
    if rubric == 'quality-v2':
        payload['checks'] = spec.model_dump(exclude_defaults=True)
    text, _ = client.generate(json.dumps(payload), judge=True, schema=schema)
    try:
        scores = schema.model_validate_json(text)
    except ValueError:
        raise ProviderError('Invalid judge response.') from None
    result = {**scores.model_dump(), 'overall': overall_score(scores, rubric),
              'rubric_version': rubric, 'judge_model': client.settings.gemini_judge_model,
              'reference_provided': reference is not None,
              'verification_status': 'reference_based_judgment' if reference is not None else 'unverified_judgment'}
    if rubric == 'quality-v2':
        result.update(score_weights=dict(WEIGHTS), **check_answer(answer, spec))
    return result
