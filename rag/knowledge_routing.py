"""Query-only routing promoted from the validated hybrid experiment.

No evaluation labels or experiment runtime imports are used by the application.
"""


import re


PARAMS = {"max_hits": 4, "body_char_budget": 2000}


def union_size(intervals):
    end = -1
    total = 0
    for start, stop in sorted(intervals):
        total += max(0, stop-max(start, end))
        end = max(end, stop)
    return total


def number(text):
    if text.isdigit():
        return int(text)
    digits = dict(zip('零〇一二三四五六七八九', [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    total = current = 0
    for char in text:
        if char in digits:
            current = digits[char]
        elif char in '十百千':
            total += (current or 1) * {'十': 10, '百': 100, '千': 1000}[char]
            current = 0
        else:
            raise ValueError('Unsupported numeral')
    return total + current


def pack(candidates):
    """Whole source spans, bounded context, no duplicate/mostly overlapping hits."""
    selected = []
    for hit in candidates:
        overlaps = [(max(hit['start'], old['start']), min(hit['end'], old['end']))
                    for old in selected if old['source'] == hit['source']]
        overlap = union_size([(a, z) for a, z in overlaps if z > a])
        if overlap >= .8 * (hit['end']-hit['start']):
            continue
        if sum(len(old['content']) for old in selected)+len(hit['content']) > PARAMS['body_char_budget']:
            continue
        selected.append(hit)
        if len(selected) == PARAMS['max_hits']:
            break
    return selected


def route_accounting(query, documents):
    numerals = r'[0-9零〇一二三四五六七八九十百千]+'
    families = {str(number(n)) for n in re.findall(r'第('+numerals+r')号', query)}
    years = {int(y) for y in re.findall(r'(?<!\d)((?:19|20)\d{2})(?!\d)', query)}
    articles = {number(n) for n in re.findall(r'第('+numerals+r')条', query)}
    # Derive aliases exclusively from corpus titles, not evaluation labels.
    aliases = {}
    for doc in documents:
        alias = re.sub(r'^企业会计准则(?:第\d+号)?[—－-]+', '', doc['title'])
        alias = re.sub(r'[（(]\d{4}[）)]$', '', alias)
        aliases[alias] = doc['standard_id'].lstrip('0') or '0'
    matches = [alias for alias in aliases if alias in query]
    # A named standard outranks an incidental mention of another document title
    # (e.g. "租赁准则 ... 现金流量表" refers to the lease standard).
    named = [alias for alias in matches if (alias if alias.endswith('准则') else alias+'准则') in query]
    matches = named or matches
    if not families and matches:
        longest = max(map(len, matches))
        families = {aliases[alias] for alias in matches if len(alias) == longest}
    docs = [d for d in documents if (d['standard_id'].lstrip('0') or '0') in families
            and (not years or d['version_year'] in years)]
    # Multiple standards/clauses can have ambiguous pairings; do not cross-product them.
    exact = bool(len(families) == 1 and years and len(articles) == 1)
    return {'families': sorted(families), 'years': sorted(years), 'articles': sorted(articles),
            'sources': [d['id']+'.md' for d in docs] if families else None,
            'exact_enabled': exact, 'version_unspecified': bool(families and not years),
            'unmatched_explicit_scope': bool(families and not docs)}
