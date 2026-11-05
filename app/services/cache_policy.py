"""A deliberately narrow lexical veto, not an entailment or freshness proof."""
import re
import unicodedata

# Keep every word and symbol, including Unicode comparisons/currencies and code
# delimiters. Quoted literals retain their spelling and internal whitespace.
_TOKEN = re.compile(r'''"[^"\n]*"|'[^'\n]*'|[^\W\d]\w*(?:['’][^\W\d]\w*)*|\d+(?:\.\d+)?|==|!=|<=|>=|[^\w\s]''', re.UNICODE)


def content_signature(text):
    tokens = _TOKEN.findall(unicodedata.normalize('NFC', text))
    # One explicit, case-sensitive geographic rewrite; do not drop "city"
    # globally (Mexico City and Mexico must remain different).
    return tuple(token for index, token in enumerate(tokens)
                 if not (token == 'city' and index and tokens[index - 1] == 'capital'))


def allows_semantic_reuse(source, query):
    # Whitespace and indentation in multiline code can change its meaning.
    if any(char in source or char in query for char in ('\n', '\r')):
        return False
    signature = content_signature(source)
    return bool(signature) and signature == content_signature(query)
