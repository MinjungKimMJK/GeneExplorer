"""User-triggered g:Profiler interpretation with recorded provenance."""
from datetime import datetime, timezone
import requests


def profile_genes(query, background, sources=('GO:BP', 'GO:MF', 'KEGG'), post=None):
    query, background = sorted(set(query)), sorted(set(background))
    if len(query) < 3 or not set(query) <= set(background):
        raise ValueError('Use at least three query genes contained in the analysis background.')
    if not sources or not set(sources) <= {'GO:BP', 'GO:MF', 'GO:CC', 'KEGG'}:
        raise ValueError('Select supported GO or KEGG sources.')
    payload = dict(organism='hsapiens', query=query, background=background, sources=list(sources),
                   domain_scope='custom', significance_threshold_method='fdr', user_threshold=.05,
                   all_results=True, ordered=False, no_evidences=False)
    response = (post or requests.post)('https://biit.cs.ut.ee/gprofiler/api/gost/profile/', json=payload, timeout=45)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data.get('result'), list) or not isinstance(data.get('meta'), dict):
        raise ValueError('The enrichment service returned an unexpected response.')
    return {'requested_utc': datetime.now(timezone.utc).isoformat(), 'request': payload,
            'meta': data['meta'], 'result': data['result'],
            'interpretation': 'Returned p_value is service-adjusted. FDR was requested; correction does not extend across separately run clusters. HGNC-related enrichment is not independent performance evidence.'}
