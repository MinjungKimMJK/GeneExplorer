"""Result-preserving, memory-bounded structural graph construction.

This module changes only how directed top-k neighbors are computed.  It does
not change A, K, weights, tie breaking, top-k, or maximum symmetrization.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from ._frozen_core import off_diagonal


def sparse_bytes(matrix: sparse.spmatrix) -> int:
    matrix = sparse.csr_matrix(matrix)
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def blockwise_topk_product(
    left: sparse.spmatrix,
    right: sparse.spmatrix,
    k: int,
    gene_ids: list[str],
    block_size: int = 256,
):
    """Return top-k(left @ right) with the frozen engine's exact semantics.

    Selection is directed and positive, boundary ties use canonical gene ID,
    and the selected graph is symmetrized by maximum.  Only one row block of
    the unsparsified product is materialized at a time.
    """
    if not isinstance(k, int) or k < 1:
        raise ValueError("top_k must be a positive integer")
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    left = sparse.csr_matrix(left, dtype=np.float64)
    right = sparse.csr_matrix(right, dtype=np.float64)
    n = len(gene_ids)
    if left.shape[0] != n or right.shape[1] != n or left.shape[1] != right.shape[0]:
        raise ValueError("Invalid product dimensions")
    if len(set(gene_ids)) != n:
        raise ValueError("Gene IDs must be unique")

    id_rank = {gid: i for i, gid in enumerate(sorted(gene_ids))}
    rank = np.array([id_rank[gid] for gid in gene_ids])
    rr: list[int] = []
    cc: list[int] = []
    vv: list[float] = []
    peak_block_sparse_bytes = 0
    peak_block_nnz = 0

    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        block = (left[start:stop] @ right).tocsr()
        block.sum_duplicates()
        block.sort_indices()
        peak_block_sparse_bytes = max(peak_block_sparse_bytes, sparse_bytes(block))
        peak_block_nnz = max(peak_block_nnz, int(block.nnz))
        if block.data.size and (
            not np.isfinite(block.data).all() or (block.data < 0).any()
        ):
            raise ValueError("Affinity must have finite nonnegative weights")

        for local_row in range(block.shape[0]):
            global_row = start + local_row
            lo, hi = block.indptr[local_row : local_row + 2]
            cols = block.indices[lo:hi]
            values = block.data[lo:hi]
            keep = (cols != global_row) & (values > 0)
            cols, values = cols[keep], values[keep]
            order = np.lexsort((rank[cols], -values))[:k]
            rr.extend([global_row] * len(order))
            cc.extend(cols[order].tolist())
            vv.extend(values[order].tolist())

    directed = sparse.csr_matrix((vv, (rr, cc)), shape=(n, n), dtype=np.float64)
    graph = off_diagonal(directed.maximum(directed.T))
    audit = {
        "block_size": block_size,
        "blocks": int((n + block_size - 1) // block_size),
        "peak_unsparsified_block_nnz": peak_block_nnz,
        "peak_unsparsified_block_sparse_bytes": peak_block_sparse_bytes,
        "directed_topk_nnz": int(directed.nnz),
        "final_graph_nnz": int(graph.nnz),
        "final_graph_sparse_bytes": sparse_bytes(graph),
    }
    return graph, audit


def structural_graphs_blockwise(A, K, k, gene_ids, block_size=256):
    """Build family and hierarchy graphs while reusing the cached A @ K."""
    A = sparse.csr_matrix(A, dtype=np.float64)
    K = sparse.csr_matrix(K, dtype=np.float64)
    AK = (A @ K).tocsr()
    family, family_audit = blockwise_topk_product(
        A, A.T, k, gene_ids, block_size=block_size
    )
    hierarchy, hierarchy_audit = blockwise_topk_product(
        AK, A.T, k, gene_ids, block_size=block_size
    )
    return family, hierarchy, AK, {
        "family_only": family_audit,
        "hierarchy": hierarchy_audit,
        "A_sparse_bytes": sparse_bytes(A),
        "K_sparse_bytes": sparse_bytes(K),
        "AK_sparse_bytes": sparse_bytes(AK),
    }
