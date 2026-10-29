"""Pure mapping, affinity, graph, and explanation operations.

No downloads, installation, UI, or file writes occur at import time.
Affinities are symmetric/nonnegative, not asserted positive semidefinite.
"""
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
import igraph as ig
import leidenalg


def tokens(value):
    return {v.strip() for v in (value or '').replace(';', '|').split('|') if v.strip()}


@dataclass
class Reference:
    genes: dict
    exact: dict
    historical: dict
    family_sizes: dict
    family_ids: tuple

    @classmethod
    def from_rows(cls, rows):
        genes, exact, historical = {}, {}, defaultdict(set)
        sizes = Counter()
        for row in rows:
            if row.get('status', 'Approved') != 'Approved':
                continue
            gid, symbol = row['hgnc_id'], row['symbol'].strip()
            if not gid or not symbol or gid in genes or symbol.upper() in exact:
                raise ValueError('Missing or duplicate approved HGNC ID/symbol')
            families = sorted(tokens(row.get('gene_group_id', '')))
            genes[gid] = {'symbol': symbol, 'families': families}
            exact[symbol.upper()] = gid
            sizes.update(families)
            for old in tokens(row.get('prev_symbol', '')) | tokens(row.get('alias_symbol', '')):
                historical[old.upper()].add(gid)
        return cls(genes, exact, dict(historical), dict(sizes), tuple(sorted(sizes)))

    @classmethod
    def read(cls, path):
        with Path(path).open(encoding='utf-8-sig', newline='') as f:
            return cls.from_rows(csv.DictReader(f, delimiter='\t'))

    def resolve(self, symbol, overrides=None):
        original = str(symbol).strip()
        key = original.upper()
        status, hits = 'unmapped', []
        if key in self.exact:
            status, hits = 'exact', [self.exact[key]]
        elif overrides and original in overrides:
            target = overrides[original].upper()
            if target not in self.exact:
                raise ValueError(f'Override target is not approved: {target}')
            status, hits = 'evidence_override', [self.exact[target]]
        else:
            hits = sorted(self.historical.get(key, set()))
            status = 'historical_unique' if len(hits) == 1 else 'ambiguous' if hits else 'unmapped'
        gid = hits[0] if len(hits) == 1 else ''
        gene = self.genes.get(gid, {})
        return {'input_symbol': original, 'mapping_status': status, 'hgnc_id': gid,
                'approved_symbol': gene.get('symbol', ''),
                'family_ids': '|'.join(gene.get('families', [])),
                'candidate_hgnc_ids': '|'.join(hits)}


def incidence(reference, gene_ids, normalization='reference_inverse_sqrt'):
    if len(set(gene_ids)) != len(gene_ids):
        raise ValueError('Duplicate gene IDs in matrix index')
    if normalization not in {'reference_inverse_sqrt', 'reference_inverse', 'none'}:
        raise ValueError('Unsupported normalization')
    col = {f: j for j, f in enumerate(reference.family_ids)}
    rows, cols, values = [], [], []
    for i, gid in enumerate(gene_ids):
        for family in reference.genes[gid]['families']:
            size = reference.family_sizes[family]
            weight = 1/np.sqrt(size) if normalization == 'reference_inverse_sqrt' else 1/size if normalization == 'reference_inverse' else 1.
            rows.append(i); cols.append(col[family]); values.append(weight)
    return sparse.csr_matrix((values, (rows, cols)), shape=(len(gene_ids), len(col)), dtype=np.float64)


def hierarchy_kernel(family_ids, closure_rows, alpha=.5, max_distance=3, family_only=False):
    if not 0 < alpha <= 1 or max_distance < 0 or int(max_distance) != max_distance:
        raise ValueError('Invalid hierarchy parameters')
    if len(set(family_ids)) != len(family_ids):
        raise ValueError('Duplicate families')
    index = {f: j for j, f in enumerate(family_ids)}
    pairs, audit = {}, Counter()
    for row in closure_rows:
        p, c = str(row['parent_fam_id']), str(row['child_fam_id'])
        d_float = float(row['distance'])
        if not np.isfinite(d_float) or d_float < 0 or not d_float.is_integer():
            raise ValueError('Closure distance must be a nonnegative integer')
        d = int(d_float)
        if p == c:
            if d != 0:
                raise ValueError('Nonzero distance for self relation')
            audit['self_rows_ignored'] += 1
            continue
        if p not in index or c not in index:
            audit['rows_without_reference_endpoints'] += 1
            if d == 0: audit['invalid_zero_distance_outside_reference_ignored'] += 1
            continue
        if d == 0:
            raise ValueError('Zero distance between different active families')
        if d > max_distance:
            audit['beyond_max_distance'] += 1
            continue
        key = tuple(sorted((index[p], index[c])))
        if key in pairs:
            audit['repeated_unordered_pairs'] += 1
        pairs[key] = min(d, pairs.get(key, d))
    n = len(index)
    rr, cc, vv = list(range(n)), list(range(n)), [1.]*n
    if not family_only:
        for (i,j), d in sorted(pairs.items()):
            rr.extend([i,j]); cc.extend([j,i]); vv.extend([alpha**d]*2)
    kernel = sparse.csr_matrix((vv,(rr,cc)), shape=(n,n))
    audit['off_diagonal_pairs_used'] = 0 if family_only else len(pairs)
    return kernel, dict(audit)


def off_diagonal(matrix):
    result = sparse.csr_matrix(matrix, dtype=np.float64).copy()
    result.sum_duplicates(); result.setdiag(0); result.eliminate_zeros(); result.sort_indices()
    if result.data.size and (not np.isfinite(result.data).all() or (result.data < 0).any()):
        raise ValueError('Affinity must have finite nonnegative weights')
    return result


def sparsify(matrix, k, gene_ids):
    """Directed positive top-k, stable ID tie break, then symmetric maximum.

    The symmetric graph may have degree > k. IDs must be unique and canonical.
    """
    if not isinstance(k, int) or k < 1:
        raise ValueError('top_k must be a positive integer')
    mat = off_diagonal(matrix)
    if mat.shape != (len(gene_ids),len(gene_ids)) or len(set(gene_ids)) != len(gene_ids):
        raise ValueError('Invalid gene index')
    id_rank = {gid: i for i,gid in enumerate(sorted(gene_ids))}
    rank = np.array([id_rank[g] for g in gene_ids])
    rr, cc, vv = [], [], []
    for i in range(mat.shape[0]):
        start, end = mat.indptr[i:i+2]
        cols, vals = mat.indices[start:end], mat.data[start:end]
        order = np.lexsort((rank[cols], -vals))[:k]
        rr.extend([i]*len(order)); cc.extend(cols[order]); vv.extend(vals[order])
    out = sparse.csr_matrix((vv,(rr,cc)), shape=mat.shape)
    return off_diagonal(out.maximum(out.T))


def expression_graph(expression, gene_ids, top_k, block_size=256):
    """Positive Pearson correlation of log2 processed microarray intensities.

    Blocked multiplication avoids allocating the full square dense matrix.
    A constant row becomes an isolate; negative correlations are not edges.
    """
    expr = np.asarray(expression, dtype=float)
    if expr.ndim != 2 or expr.shape[0] != len(gene_ids) or expr.shape[1] < 3:
        raise ValueError('Expression requires >=3 samples and one row per gene')
    if not np.isfinite(expr).all() or np.any(expr <= 0):
        raise ValueError('This microarray transform requires positive finite values')
    z = np.log2(expr); z -= z.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(z, axis=1)
    nonconstant = norms > 1e-12
    z[nonconstant] /= norms[nonconstant,None]; z[~nonconstant] = 0
    rr, cc, vv = [], [], []
    ranks = {g:i for i,g in enumerate(sorted(gene_ids))}
    rank = np.array([ranks[g] for g in gene_ids])
    for start in range(0,len(gene_ids),block_size):
        block = np.clip(z[start:start+block_size] @ z.T, 0, 1)
        for offset, weights in enumerate(block):
            i = start+offset; weights[i] = 0
            cols = np.flatnonzero(weights > 0)
            choice = cols[np.lexsort((rank[cols], -weights[cols]))[:top_k]]
            rr.extend([i]*len(choice)); cc.extend(choice); vv.extend(weights[choice])
    mat = sparse.csr_matrix((vv,(rr,cc)), shape=(len(gene_ids),len(gene_ids)))
    return off_diagonal(mat.maximum(mat.T)), {'constant_rows':int((~nonconstant).sum()),'samples':expr.shape[1]}


def to_graph(matrix):
    mat = off_diagonal(matrix)
    if (mat-mat.T).nnz and np.max(np.abs((mat-mat.T).data)) > 1e-12:
        raise ValueError('Graph must be symmetric')
    upper = sparse.triu(mat, k=1, format='coo')
    graph = ig.Graph(n=mat.shape[0], edges=list(zip(upper.row.tolist(),upper.col.tolist())), directed=False)
    graph.es['weight'] = upper.data.tolist()
    return graph


def cluster(matrix, gene_ids, resolution=1., seed=42, iterations=2):
    if resolution <= 0 or not np.isfinite(resolution) or iterations < 1:
        raise ValueError('Invalid Leiden parameters')
    if len(gene_ids) == 0:
        return np.array([], dtype=int)
    if list(gene_ids) != sorted(gene_ids):
        order = np.argsort(gene_ids)
        ordered = sparse.csr_matrix(matrix)[order][:,order]
        labels = cluster(ordered, [gene_ids[i] for i in order], resolution, seed, iterations)
        result = np.empty(len(gene_ids), dtype=int); result[order] = labels
        return result
    graph = to_graph(matrix)
    if graph.ecount() == 0:
        communities = [[i] for i in range(len(gene_ids))]
    else:
        partition = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition,
            weights='weight', resolution_parameter=resolution, seed=seed, n_iterations=iterations)
        communities = list(partition)
    # A zero-degree gene has no support for co-membership: keep it singleton.
    isolated = {i for i,d in enumerate(graph.degree()) if d == 0}
    communities = [[i for i in members if i not in isolated] for members in communities]
    communities = [m for m in communities if m] + [[i] for i in isolated]
    communities.sort(key=lambda members: tuple(sorted(gene_ids[i] for i in members)))
    labels = np.empty(len(gene_ids), dtype=int)
    for label, members in enumerate(communities):
        labels[members] = label
    return labels


def graph_summary(matrix, labels):
    graph = to_graph(matrix)
    sizes = Counter(map(int,labels)); values = list(sizes.values())
    return {'genes':graph.vcount(), 'edges':graph.ecount(),
            'isolates':sum(d == 0 for d in graph.degree()), 'connected_components':len(graph.connected_components()),
            'clusters':len(sizes),'singleton_clusters':sum(v == 1 for v in values),
            'clusters_ge4':sum(v >= 4 for v in values), 'largest_cluster':max(values,default=0),
            'median_cluster_size':float(np.median(values)) if values else 0.,
            'sparse_bytes':int(matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes)}


def recommend(reference, kernel, seed_ids, excluded_ids=(), normalization='reference_inverse_sqrt'):
    """Unnormalized A K A^T x with exact same-family/hierarchy/seed parts."""
    if not seed_ids or len(set(seed_ids)) != len(seed_ids):
        raise ValueError('Seeds must be nonempty and unique')
    universe = sorted(reference.genes)
    idx = {g:i for i,g in enumerate(universe)}
    A = incidence(reference, universe, normalization)
    seed_rows = A[[idx[s] for s in seed_ids]]
    per_seed = (A @ kernel @ seed_rows.T).toarray()
    same = (A @ seed_rows.T).toarray()
    other = per_seed-same
    if np.min(other, initial=0) < -1e-12:
        raise ValueError('Invalid hierarchy score decomposition')
    excluded = set(excluded_ids) | set(seed_ids)
    output = []
    for i,gid in enumerate(universe):
        total = float(per_seed[i].sum())
        if gid in excluded or total <= 0:
            continue
        output.append({'hgnc_id':gid,'symbol':reference.genes[gid]['symbol'],'score':total,
            'same_family_score':float(same[i].sum()),'hierarchy_score':float(other[i].sum()),
            **{f'seed_{s}':float(per_seed[i,j]) for j,s in enumerate(seed_ids)}})
    return sorted(output, key=lambda r:(-r['score'],r['hgnc_id']))


def explain_pair(reference, kernel, candidate_id, seed_id, normalization='reference_inverse_sqrt'):
    A = incidence(reference,[candidate_id,seed_id],normalization)
    left, right = A.getrow(0), A.getrow(1)
    result = []
    for f,wf in zip(left.indices,left.data):
        for e,we in zip(right.indices,right.data):
            weight = float(kernel[f,e])
            if weight:
                result.append({'candidate_id':candidate_id,'seed_id':seed_id,
                    'candidate_family':reference.family_ids[f],'seed_family':reference.family_ids[e],
                    'relation':'same_family' if f == e else 'hierarchy',
                    'kernel_weight':weight,'contribution':float(wf*weight*we)})
    return result
