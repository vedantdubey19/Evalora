import json

from pydantic import BaseModel, Field

from app.services.llm_client import ProviderError


class Scores(BaseModel):
    correctness: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    reasoning: str = Field(min_length=1, max_length=8000)


def evaluate(client, question, answer, reference):
    prompt = json.dumps({'question': question, 'answer': answer, 'reference': reference})
    text, _ = client.generate(prompt, judge=True, schema=Scores)
    try:
        scores = Scores.model_validate_json(text)
    except ValueError:
        raise ProviderError('Invalid judge response.') from None
    return {
        **scores.model_dump(),
        'overall': round((scores.correctness + scores.relevance + scores.clarity) / 3, 2),
        'judge_model': client.settings.gemini_judge_model,
        'reference_provided': reference is not None,
    }
