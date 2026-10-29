"""Application adapters for the frozen coauthor computations.

No UI, installations or network requests occur here. Uploaded expression must
already be normalized log2/logCPM; raw-count differential expression stays in R.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from io import BytesIO, StringIO
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata

from . import __version__
from ._frozen_core import Reference, incidence, hierarchy_kernel, cluster, graph_summary, recommend, explain_pair
from ._frozen_graph import structural_graphs_blockwise

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('Expression correlation', 'Family-only', 'Hierarchy')
SLUG = {'Expression correlation': 'expression_correlation', 'Family-only': 'family_only',
        'Hierarchy': 'hierarchy', 'Direct DE': 'direct_de', 'Degree-only': 'degree_only'}


@dataclass(frozen=True)
class Settings:
    alpha: float = 0.5
    max_distance: int = 3
    top_k: int = 25
    resolution: float = 1.0
    seed: int = 42
    iterations: int = 2
    mixture: float = 0.5
    make_layout: bool = False

    def validate(self):
        if not np.isfinite([self.alpha, self.resolution, self.mixture]).all():
            raise ValueError('Settings must be finite.')
        if not 0 <= self.alpha <= 1 or not 0 <= self.mixture <= 1:
            raise ValueError('Decay and mixture weights must be between 0 and 1.')
        if self.max_distance not in (1, 2, 3) or self.resolution <= 0:
            raise ValueError('Use hierarchy distance 1–3 and positive Leiden resolution.')
        if not isinstance(self.top_k, int) or self.top_k < 1:
            raise ValueError('Top-k must be a positive integer.')
        if self.iterations != 2 or self.seed != 42:
            raise ValueError('The supported frozen Leiden configuration uses seed 42 and two iterations.')


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_table(payload: bytes) -> pd.DataFrame:
    try:
        text = payload.decode('utf-8-sig')
    except UnicodeDecodeError as exc:
        raise ValueError('Please save the table as UTF-8 CSV or TSV.') from exc
    if not text.strip():
        raise ValueError('The uploaded table is empty.')
    delimiter = '\t' if '\t' in text.splitlines()[0] else ','
    header = next(csv.reader(StringIO(text), delimiter=delimiter))
    header = [x.strip() for x in header]
    if len(header) != len(set(header)) or any(not x for x in header):
        raise ValueError('Every column needs a unique, nonempty header. Rename repeated sample columns.')
    frame = pd.read_csv(StringIO(text), sep=delimiter, dtype=str, keep_default_na=False)
    frame.columns = header
    if frame.empty:
        raise ValueError('The table has headers but no data rows.')
    return frame


def family_name_map(frame: pd.DataFrame) -> dict[str, str]:
    """Pair IDs/names in source order; never sort the two lists independently."""
    names = {}
    for row in frame.to_dict('records'):
        if row.get('status') != 'Approved' or not row.get('gene_group_id'):
            continue
        ids = [x.strip() for x in row['gene_group_id'].split('|')]
        labels = [x.strip() for x in row.get('gene_group', '').split('|')]
        if len(ids) != len(labels) or any(not x for x in ids + labels):
            raise ValueError(f'Family IDs/names do not align for {row.get("symbol", "a gene")}.')
        for fid, name in zip(ids, labels):
            if fid in names and names[fid] != name:
                raise ValueError(f'Conflicting family names for ID {fid}.')
            names[fid] = name
    return names


def load_reference(hgnc_bytes=None, closure_bytes=None):
    hgnc_bytes = hgnc_bytes if hgnc_bytes is not None else (ROOT/'hgnc_complete_set.txt').read_bytes()
    closure_bytes = closure_bytes if closure_bytes is not None else (ROOT/'hierarchy_closure.csv').read_bytes()
    hgnc = read_table(hgnc_bytes)
    required = {'hgnc_id', 'symbol', 'status', 'gene_group_id', 'gene_group'}
    if not required <= set(hgnc):
        raise ValueError('HGNC reference requires: '+', '.join(sorted(required)))
    reference = Reference.from_rows(hgnc.to_dict('records'))
    if not reference.family_ids:
        raise ValueError('The reference has no approved family-annotated genes.')
    names = family_name_map(hgnc)
    # Seven parent-only names supplement the frozen display dictionary only.
    if digest(hgnc_bytes) == '0e0e04c1a5dcfe0f02a9e448fc808e5407d2a8f3cc359b7f6856362bb6a93dcb':
        extra = pd.read_csv(ROOT/'data/reference/family_id_name_mapping.csv', dtype=str)
        for row in extra.itertuples():
            if row.family_id in names and names[row.family_id] != row.family_name:
                raise ValueError('Frozen family display dictionary conflicts with the gene reference.')
            names[row.family_id] = row.family_name
    closure = read_table(closure_bytes)
    if not {'parent_fam_id', 'child_fam_id', 'distance'} <= set(closure):
        raise ValueError('Hierarchy CSV requires parent_fam_id, child_fam_id and distance.')
    return reference, names, closure.to_dict('records'), {'hgnc_complete_set.txt': hgnc_bytes, 'hierarchy_closure.csv': closure_bytes}


def map_genes(frame, identifier, reference, allow_aliases=False, display_column=None):
    if identifier not in frame:
        raise ValueError('Choose a gene identifier column.')
    audit, mapped, seen = [], [], set()
    display = None
    if display_column:
        display = pd.to_numeric(frame[display_column], errors='coerce').to_numpy(float)
        if not np.isfinite(display).all():
            raise ValueError('The display-value column must contain finite numeric values in every row.')
    for position, token in enumerate(frame[identifier]):
        token = str(token).strip()
        if token.upper().startswith('HGNC:'):
            gid = token.upper()
            status = 'hgnc_id' if gid in reference.genes else 'unmapped'
        else:
            record = reference.resolve(token)
            gid, status = record['hgnc_id'], record['mapping_status']
            if status == 'historical_unique' and not allow_aliases:
                gid, status = '', 'alias_not_enabled'
        if gid not in reference.genes:
            gid = ''
        reason = ''
        if not gid:
            reason = status
        elif not reference.genes[gid]['families']:
            reason = 'no_family_annotation'
        elif gid in seen:
            raise ValueError(f'Multiple rows map to {gid}. Resolve duplicates before uploading; rows are not silently averaged.')
        audit.append({'input_row': position+1, 'input_identifier': token, 'hgnc_id': gid,
                      'mapping_status': status, 'included': bool(gid and not reason), 'exclusion_reason': reason})
        if gid and not reason:
            seen.add(gid)
            mapped.append({'hgnc_id': gid, 'symbol': reference.genes[gid]['symbol'],
                           'family_ids': '|'.join(reference.genes[gid]['families']),
                           'input_position': position, 'display_value': float(display[position]) if display is not None else np.nan})
    if not mapped:
        raise ValueError('No unique approved, family-annotated genes remain after mapping.')
    genes = pd.DataFrame(mapped).sort_values('hgnc_id').reset_index(drop=True)
    if len(genes) > 15000:
        raise ValueError('This interactive deployment supports up to 15,000 mapped genes. Use the offline projects for larger inputs.')
    return genes, pd.DataFrame(audit)


def numeric_matrix(frame, columns):
    if len(columns) < 3:
        raise ValueError('Expression comparison requires at least three sample columns, not group means or fold changes.')
    x = frame[columns].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    if not np.isfinite(x).all():
        raise ValueError('Expression samples must be finite numeric values without blanks.')
    return x


def load_example(dataset):
    if dataset not in ('GSE74000', 'GSE241735'):
        raise ValueError('Unknown example dataset.')
    manifest = json.loads((ROOT/'docs/SOURCE_PROVENANCE.json').read_text(encoding='utf-8'))
    # Do not attach frozen significance tables to changed local inputs.
    for item in manifest['files']:
        target = item['destination']
        if target in ['hgnc_complete_set.txt', 'hierarchy_closure.csv', 'APAP_healthy_apap_human(fold_change).csv'] or target.startswith('data/'):
            if digest((ROOT/target).read_bytes()) != item['sha256']:
                raise ValueError(f'Frozen input changed: {target}. Restore the release file or use Custom mode.')
    ref, names, closure, files = load_reference()
    folder = ROOT/'data'/dataset
    if dataset == 'GSE74000':
        gene_file, exp_file = 'genes.csv', 'expression_log2.csv'
    else:
        gene_file, exp_file = 'gene_universe.csv', 'discovery_logcpm.csv'
    for name in [gene_file, exp_file]:
        files[name] = (folder/name).read_bytes()
    genes, audit = map_genes(read_table(files[gene_file]), 'hgnc_id', ref)
    exp = read_table(files[exp_file]).set_index('hgnc_id').loc[genes.hgnc_id]
    x = numeric_matrix(exp, exp.columns.tolist())
    raw_y, validation = None, {}
    if dataset == 'GSE241735':
        for name in ['de_ICC13_7_24h.csv', 'de_MG69.csv', 'de_MG212.csv']:
            files[name] = (folder/name).read_bytes()
            table = pd.read_csv(BytesIO(files[name])).set_index('hgnc_id').loc[genes.hgnc_id]
            if name == 'de_ICC13_7_24h.csv':
                raw_y = table.signed_stat.to_numpy(float)
                genes['display_value'] = table.logFC.to_numpy(float)
            else:
                validation[name.removeprefix('de_').removesuffix('.csv')] = table
        note = 'Frozen common expressed-gene universe uses filters from ICC13-7, MG69 and MG212. PDX response statistics do not enter the scores. This is a conditional application case, not untouched external validation.'
    else:
        # Historical summaries are display-only; the expression baseline above
        # uses the frozen five-sample matrix, not means/ratios from this table.
        legacy_name = 'APAP_healthy_apap_human(fold_change).csv'
        files[legacy_name] = (ROOT/legacy_name).read_bytes()
        historical = pd.read_csv(BytesIO(files[legacy_name]), dtype=str, keep_default_na=False)
        displayed, audit = map_genes(historical, 'Symbol', ref, display_column='fold change')
        if set(displayed.hgnc_id) != set(genes.hgnc_id):
            raise ValueError('Historical exact mapping no longer matches the frozen 2,523-gene universe.')
        genes['display_value'] = displayed.set_index('hgnc_id').loc[genes.hgnc_id, 'display_value'].to_numpy()
        note = 'Frozen historical exact-mapped 2,523-gene comparison; three APAP-ALF samples and two controls. Original 3,713-symbol selection rationale remains uncertain. The legacy count column is a control mean, not read count.'
    return dict(dataset=dataset, mode='ranking' if raw_y is not None else 'clustering', reference=ref,
                family_names=names, closure=closure, genes=genes, mapping=audit, expression=x,
                samples=list(exp.columns), raw_y=raw_y, validation=validation, source_files=files,
                display_label='Discovery log2 fold change' if raw_y is not None else 'APAP fold change (case mean / control mean)', note=note,
                preprocessing={'source': 'frozen prepared inputs', 'identifier': 'hgnc_id', 'aliases': False})


def load_custom(payload, identifier, samples=(), display_column=None, allow_aliases=False,
                score_payload=None, hgnc_bytes=None, closure_bytes=None):
    ref, names, closure, files = load_reference(hgnc_bytes, closure_bytes)
    frame = read_table(payload)
    genes, audit = map_genes(frame, identifier, ref, allow_aliases, display_column)
    x = numeric_matrix(frame, list(samples))[genes.input_position] if samples else None
    raw_y = None
    files['uploaded_genes.csv'] = payload
    if score_payload is not None:
        scores = read_table(score_payload)
        if not {'hgnc_id', 'signed_stat'} <= set(scores):
            raise ValueError('Ranking input requires hgnc_id and signed_stat (signed sqrt of discovery QL F).')
        if scores.hgnc_id.duplicated().any():
            raise ValueError('Discovery score IDs must be unique.')
        if not set(genes.hgnc_id) <= set(scores.hgnc_id):
            raise ValueError('Every included gene needs a discovery statistic; missing scores are not replaced by zero.')
        raw_y = pd.to_numeric(scores.set_index('hgnc_id').loc[genes.hgnc_id, 'signed_stat'], errors='coerce').to_numpy(float)
        files['discovery_scores.csv'] = score_payload
    return dict(dataset='Custom', mode='ranking' if raw_y is not None else 'clustering', reference=ref,
                family_names=names, closure=closure, genes=genes, mapping=audit, expression=x,
                samples=list(samples), raw_y=raw_y, validation={}, source_files=files,
                display_label=display_column, note='User-provided normalized expression. No automatic log transform, raw-count DE, PDX filtering or biological validation is performed.',
                preprocessing={'identifier': identifier, 'sample_columns': list(samples), 'display_column': display_column,
                               'aliases': allow_aliases, 'expression_scale': 'user-supplied log2/logCPM; no automatic transform'})


def expression_graph(values, ids, k, block_size=256):
    """Frozen positive Pearson construction on already transformed profiles."""
    z = np.asarray(values, dtype=np.float64).copy()
    z -= z.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(z, axis=1)
    usable = norms > 1e-12
    z[usable] /= norms[usable, None]
    z[~usable] = 0
    rr, cc, vv = [], [], []
    ranks = np.arange(len(ids))  # callers sort canonical HGNC IDs first
    for start in range(0, len(ids), block_size):
        block = np.clip(z[start:start+block_size] @ z.T, 0, 1)
        for offset, weights in enumerate(block):
            row = start+offset
            weights[row] = 0
            cols = np.flatnonzero(weights > 0)
            selected = cols[np.lexsort((ranks[cols], -weights[cols]))[:k]]
            rr.extend([row]*len(selected)); cc.extend(selected); vv.extend(weights[selected])
    w = sparse.csr_matrix((vv, (rr, cc)), shape=(len(ids), len(ids)))
    return w.maximum(w.T).tocsr(), int((~usable).sum())


def ranking_scores(graphs, raw_y):
    raw_y = np.asarray(raw_y, float)
    norm = np.linalg.norm(raw_y)
    if not np.isfinite(raw_y).all() or not np.isfinite(norm) or norm <= 0:
        raise ValueError('Discovery statistics must be finite and not all zero.')
    return raw_y/norm


def compute(bundle, settings: Settings):
    settings.validate()
    ids = bundle['genes'].hgnc_id.tolist()
    ref = bundle['reference']
    if bundle['mode'] == 'ranking':
        # Match RNA-seq project's Python math.sqrt operation exactly.
        cols = {f: j for j, f in enumerate(ref.family_ids)}
        rr, cc, vv = [], [], []
        for i, gid in enumerate(ids):
            for fid in ref.genes[gid]['families']:
                rr.append(i); cc.append(cols[fid]); vv.append(1/math.sqrt(ref.family_sizes[fid]))
        a = sparse.csr_matrix((vv, (rr, cc)), shape=(len(ids), len(cols)))
    else:
        a = incidence(ref, ids)
    k, kernel_audit = hierarchy_kernel(ref.family_ids, bundle['closure'], alpha=settings.alpha or .5,
                                      max_distance=settings.max_distance, family_only=settings.alpha == 0)
    family, hierarchy, _, audits = structural_graphs_blockwise(a, k, settings.top_k, ids)
    graphs = {'Family-only': family, 'Hierarchy': hierarchy}
    constant = None
    if bundle['expression'] is not None:
        graphs['Expression correlation'], constant = expression_graph(bundle['expression'], ids, settings.top_k)
    graphs = {name: graphs[name] for name in METHODS if name in graphs}
    labels, structure, rankings = {}, [], None
    if bundle['mode'] == 'clustering':
        for name, graph in graphs.items():
            labels[name] = cluster(graph, ids, settings.resolution, settings.seed, settings.iterations)
            structure.append({'method': name, **graph_summary(graph, labels[name])})
    else:
        y = ranking_scores(graphs, bundle['raw_y'])
        rankings = bundle['genes'][['hgnc_id', 'symbol']].copy()
        scores = {'Direct DE': y}
        for name, w in graphs.items():
            totals = np.asarray(w.sum(axis=1)).ravel()
            inverse = np.zeros_like(totals); inverse[totals > 0] = 1/totals[totals > 0]
            scores[name] = (1-settings.mixture)*y+settings.mixture*np.asarray((sparse.diags(inverse) @ w) @ y).ravel()
        degree = np.asarray(hierarchy.sum(axis=1)).ravel()
        scores['Degree-only'] = np.sign(y)*rankdata(degree, method='average')/len(ids)
        for name, score in scores.items():
            stem = SLUG[name]
            rankings[stem+'_score'] = score
            for direction in ['up', 'down']:
                order = np.lexsort((np.asarray(ids), -score if direction == 'up' else score))
                rank = np.empty(len(ids), int); rank[order] = np.arange(1, len(ids)+1)
                rankings[stem+'_'+direction+'_rank'] = rank
        for name, w in graphs.items():
            degrees = np.asarray((w != 0).sum(axis=1)).ravel()
            structure.append({'method': name, 'genes': len(ids), 'edges': sparse.triu(w, k=1).nnz,
                              'isolates': int((degrees == 0).sum()), 'connected_components': sparse.csgraph.connected_components(w)[0]})
    result = dict(bundle, graphs=graphs, labels=labels, kernel=k, rankings=rankings,
                  structure=pd.DataFrame(structure), settings=asdict(settings), graph_audits=audits,
                  kernel_audit=kernel_audit, constant_expression_rows=constant, embeddings={}, enrichments={})
    if settings.make_layout and labels:
        result['embeddings']['Hierarchy'], result['embedding_metadata'] = embed_graph(hierarchy, ids)
    settings_bytes = json.dumps(asdict(settings), sort_keys=True).encode()
    input_hashes = {name: digest(value) for name, value in bundle['source_files'].items()}
    spec = {'dataset': bundle['dataset'], 'mode': bundle['mode'], 'preprocessing': bundle['preprocessing'], 'samples': bundle['samples']}
    result['run_id'] = digest(settings_bytes+json.dumps(input_hashes, sort_keys=True).encode()+json.dumps(spec, sort_keys=True).encode()+__version__.encode())[:16]
    result['created_utc'] = datetime.now(timezone.utc).isoformat()
    result['source_hashes'] = input_hashes
    return result


def embed_graph(graph, ids):
    """Visualization of active affinity profiles; isolates retain cluster rows."""
    active = np.asarray((graph != 0).sum(axis=1)).ravel() > 0
    positions = np.flatnonzero(active)
    xy = np.full((len(ids), 2), np.nan)
    metadata = {'representation': 'sparse symmetric graph affinity rows', 'isolates_omitted_from_layout': int((~active).sum()),
                'metric': 'cosine', 'random_state': 42, 'init': 'random', 'min_dist': 0.1,
                'n_jobs': 1, 'n_neighbors': min(15, max(2, len(positions)-1))}
    if len(positions) >= 4:
        from umap import UMAP
        xy[active] = UMAP(n_components=2, metric='cosine', random_state=42, init='random', min_dist=.1,
                          n_neighbors=metadata['n_neighbors'], n_jobs=1).fit_transform(graph[active][:, active])
        metadata['algorithm'] = 'UMAP'
    elif len(positions):
        # An explicitly identified small-graph display, not a UMAP result.
        angles = np.linspace(0, 2*np.pi, len(positions), endpoint=False)
        xy[active] = np.c_[np.cos(angles), np.sin(angles)]
        metadata['algorithm'] = 'small-graph circular layout (fewer than 4 nonisolates)'
    else:
        metadata['algorithm'] = 'no layout: all genes isolated'
    return pd.DataFrame({'hgnc_id': ids, 'x': xy[:, 0], 'y': xy[:, 1], 'layout_status': np.where(active, 'positioned', 'isolated')}), metadata


def external_coherence(result):
    if result['dataset'] != 'GSE74000':
        return pd.DataFrame()
    expression = pd.read_csv(ROOT/'data/GSE120652/gene_expression_log2.csv', index_col=0)
    roles = pd.read_csv(ROOT/'data/GSE120652/sample_roles.csv')
    if expression.columns.tolist() != roles['sample'].tolist():
        raise ValueError('External sample roles do not align.')
    ids = result['genes'].hgnc_id.tolist()
    covered = [i for i, g in enumerate(ids) if g in expression.index]
    x = expression.loc[[ids[i] for i in covered]].to_numpy(float)
    rows = []
    for mode in ['pooled', 'within_group_centered']:
        z = x.copy()
        groups = [np.arange(x.shape[1])] if mode == 'pooled' else [np.flatnonzero(roles.group == g) for g in sorted(set(roles.group))]
        for ix in groups: z[:, ix] -= z[:, ix].mean(axis=1, keepdims=True)
        norms = np.linalg.norm(z, axis=1); usable = norms > 1e-12
        z[usable] /= norms[usable, None]; z[~usable] = 0
        for method, labels in result['labels'].items():
            labels = labels[covered]
            _, inverse = np.unique(labels, return_inverse=True)
            sizes = np.bincount(inverse); sums = np.zeros((len(sizes), z.shape[1]))
            np.add.at(sums, inverse, z)
            self_sums = np.bincount(inverse, weights=np.einsum('ij,ij->i', z, z), minlength=len(sizes))
            pair_sums = (np.einsum('ij,ij->i', sums, sums)-self_sums)/2
            pair_counts = sizes*(sizes-1)/2; keep = sizes >= 4
            rows.append({'mode': mode, 'method': method, 'eligible_clusters': int(keep.sum()), 'eligible_genes': int(sizes[keep].sum()),
                         'macro_mean_pair_correlation': float((pair_sums[keep]/pair_counts[keep]).mean()) if keep.any() else np.nan,
                         'pair_weighted_correlation': float(pair_sums[keep].sum()/pair_counts[keep].sum()) if keep.any() else np.nan})
    return pd.DataFrame(rows)


def response_metrics(result):
    if not result['validation']:
        return pd.DataFrame()
    truth = {}
    for model, table in result['validation'].items():
        truth[model, 'down'] = ((table.FDR <= .05) & (table.logFC <= -1)).to_numpy()
        truth[model, 'up'] = ((table.FDR <= .05) & (table.logFC >= 1)).to_numpy()
    for direction in ['down', 'up']:
        truth['Both PDX', direction] = truth['MG69', direction] & truth['MG212', direction]
    rows = []
    for method, stem in SLUG.items():
        for direction in ['down', 'up']:
            col = stem+'_'+direction+'_rank'
            if col not in result['rankings']: continue
            order = np.argsort(result['rankings'][col].to_numpy())
            for model in ['MG69', 'MG212', 'Both PDX']:
                hits = truth[model, direction][order]; ntruth = hits.sum()
                ap = np.sum((np.cumsum(hits)/np.arange(1, len(hits)+1))[hits])/ntruth if ntruth else np.nan
                rows.append({'method': method, 'model': model, 'direction': direction, 'top_k': 50,
                             'hits': int(hits[:50].sum()), 'precision': float(hits[:50].mean()), 'average_precision': float(ap)})
    return pd.DataFrame(rows)


def seed_neighbors(result, seeds, outside_only=True):
    if len(seeds) > 10:
        raise ValueError('Select at most ten seeds for interactive exploration.')
    if not seeds:
        return pd.DataFrame(columns=['hgnc_id', 'symbol', 'score', 'same_family_score', 'hierarchy_score'])
    excluded = result['genes'].hgnc_id.tolist() if outside_only else []
    records = recommend(result['reference'], result['kernel'], seeds, excluded)
    return pd.DataFrame(records) if records else pd.DataFrame(columns=['hgnc_id', 'symbol', 'score', 'same_family_score', 'hierarchy_score'])


def pair_explanations(result, candidate, seed):
    records = explain_pair(result['reference'], result['kernel'], candidate, seed)
    for r in records:
        r['candidate_family_name'] = result['family_names'].get(r['candidate_family'], r['candidate_family'])
        r['seed_family_name'] = result['family_names'].get(r['seed_family'], r['seed_family'])
    return pd.DataFrame(records)


def export_run(result):
    buffer = BytesIO()
    environment = {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy', 'igraph', 'leidenalg', 'streamlit', 'umap-learn']}
    metadata = {'app_version': __version__, 'run_id': result['run_id'], 'created_utc': result['created_utc'],
                'dataset': result['dataset'], 'mode': result['mode'], 'settings': result['settings'], 'note': result['note'],
                'input_sha256': result['source_hashes'], 'environment': environment,
                'graph_audits': result['graph_audits'], 'kernel_audit': result['kernel_audit'],
                'embedding': result.get('embedding_metadata'), 'display_column': result['display_label'],
                'preprocessing': result['preprocessing'], 'sample_columns': result['samples'],
                'graph_index': 'genes.csv in canonical lexicographic HGNC-ID order'}
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('run.json', json.dumps(metadata, indent=2))
        archive.writestr('genes.csv', result['genes'].to_csv(index=False))
        archive.writestr('mapping_audit.csv', result['mapping'].to_csv(index=False))
        archive.writestr('structure.csv', result['structure'].to_csv(index=False))
        for name, value in result['source_files'].items(): archive.writestr('inputs/'+name, value)
        for method, graph in result['graphs'].items():
            stem = SLUG[method]; data = BytesIO(); sparse.save_npz(data, graph)
            archive.writestr(stem+'/graph.npz', data.getvalue())
            if method in result['labels']:
                frame = result['genes'][['hgnc_id', 'symbol']].assign(cluster=result['labels'][method])
                archive.writestr(stem+'/clusters.csv', frame.to_csv(index=False))
        if result['rankings'] is not None: archive.writestr('rankings.csv', result['rankings'].to_csv(index=False))
        for method, embedding in result['embeddings'].items(): archive.writestr(SLUG[method]+'/embedding.csv', embedding.to_csv(index=False))
        external = external_coherence(result)
        if not external.empty:
            archive.writestr('external_coherence.csv', external.to_csv(index=False))
            for name in ['gene_expression_log2.csv', 'sample_roles.csv']:
                archive.writestr('evaluation/GSE120652/'+name, (ROOT/'data/GSE120652'/name).read_bytes())
        metrics = response_metrics(result)
        if not metrics.empty: archive.writestr('response_metrics.csv', metrics.to_csv(index=False))
        for key, value in result['enrichments'].items():
            archive.writestr('enrichment/'+key+'.json', json.dumps(value, indent=2))
        archive.writestr('README.txt', 'These are outputs of the recorded application run, not new causal or therapeutic validation.\n'
                           'Install the matching application release, then use the recorded inputs/settings. Graph row order is genes.csv.\n'
                           'UMAP is an exploratory display; isolated genes remain in the partitions and are omitted from the layout.\n'
                           'Enrichment is interpreted under its saved background and live database metadata; it is not independent model evaluation.\n')
    return buffer.getvalue()
