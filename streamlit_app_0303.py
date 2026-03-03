# streamlit_app.py
# All-in-Python: HGNC hierarchy-aware gene clustering + UMAP + Text summaries 
# Requirements: pip install streamlit plotly pandas numpy scipy python-igraph leidenalg umap-learn

import os, sys, math, subprocess, hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from gprofiler import GProfiler

# (best-effort) auto-install heavy deps if missing
def _ensure(pkg):
    try:
        __import__(pkg)
    except Exception:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
        except Exception:
            pass

for p in ["igraph", "leidenalg", "umap-learn"]:
    _ensure(p)

import igraph as ig
import leidenalg
import umap

st.set_page_config(page_title="HGNC Hierarchy Explorer", layout="wide")


# -------------------------
# Helpers
# -------------------------
def _hash_df(df: pd.DataFrame) -> str:
    if df is None or len(df) == 0:
        return "empty"
    h = hashlib.md5()
    h.update(pd.util.hash_pandas_object(df, index=True).values)
    h.update("|".join(map(str, df.columns)).encode())
    return h.hexdigest()[:10]

def parse_group_ids(cell: str) -> List[str]:
    if pd.isna(cell):
        return []
    if isinstance(cell, (int, float)):
        return [str(int(cell))]
    txt = str(cell)
    parts = [p.strip() for p in txt.replace(";", "|").split("|")]
    return [p for p in parts if p]

def build_name_map(hgnc_df: pd.DataFrame) -> Dict[str, str]:
    name_map = {}
    for _, r in hgnc_df.dropna(subset=["gene_group_id"]).iterrows():
        gids = parse_group_ids(r["gene_group_id"])
        name = str(r.get("gene_group", "")).strip()
        for gid in gids:
            if gid and gid not in name_map and name:
                name_map[str(gid)] = name
    return name_map


# -------------------------
# Sidebar — Data & Params
# -------------------------
st.sidebar.header("1) Data")
UPLOAD_MODE = st.sidebar.radio("Data source", ["Upload files", "Path mode (read from folder)"], index=0)

base_path = Path("./")

hgnc_file = base_path / "hgnc_complete_set.txt"
clos_file  = base_path / "hierarchy_closure.csv"
curated_file = base_path / "APAP_healthy_apap_human(fold_change).csv"

st.sidebar.info("📊 **Example Mode Active**")
st.sidebar.caption(f"Using default HGNC & APAP datasets from repository.")

if st.sidebar.button("🚀 Run Analysis Pipeline", type="primary"):
    st.session_state['run_pipeline'] = True
    
st.sidebar.divider()
st.sidebar.header("2) Parameters")

# One-knob mode (default)
ONE_KNOB = st.sidebar.checkbox("One-knob mode (1 slider)", value=True)
if ONE_KNOB:
    LEVEL = st.sidebar.slider("Detail level", 0, 100, 50,
                              help="Left: coarser (fewer/larger clusters). Right: finer (more/smaller clusters).")
    use_exp = False
    alpha = 0.3 + 0.7*(LEVEL/100.0)              # 0.3..1.0
    lmbda = 1.0                                  # unused (power decay)
    DMAX  = int(round(1 + 3*(LEVEL/100.0)))      # 1..4
    TOPK  = max(5, min(200, int(round(50 - 35*(LEVEL/100.0)))))  # 50..15
    RESOLUTION = round(0.6 + 1.4*(LEVEL/100.0), 2)               # 0.6..2.0
    st.sidebar.caption(f"α≈{alpha:.2f} · Dmax={DMAX} · top-k={TOPK} · resolution={RESOLUTION}")

    with st.sidebar.expander("Advanced (optional)"):
        KEEP_ISOLATES = st.checkbox("Keep genes with no family membership", value=False)
        COUNT_FILTER  = st.slider("Exclude genes with count ≤ N", 0, 90, 0, step=1)
        COUNT_WEIGHTING = st.checkbox("Weight similarity by count (importance)", value=False)
        COUNT_GAIN = st.slider("Count weight gain (multiplier)", 0.0, 2.0, 0.8, step=0.1)

        SIZE_BY_COUNT = st.checkbox("Scale dot size by count (UMAP)", value=True)
        SIZE_MIN, SIZE_MAX = st.slider("Dot size range", 1, 28, (1, 18))
        COUNT_SIZE_TH = st.slider("Size threshold (count ≥ T scales up)", 1, 50, 5)
        SIZE_BASE = st.slider("Size for counts < T", 1, 10, 1)
        SIZE_SCALE_MODE = st.selectbox("Size scale above T", ["log","linear"], index=0)
else:
    use_exp   = st.sidebar.checkbox("Use exponential decay exp(-lambda·d)", value=False)
    alpha     = st.sidebar.number_input("alpha (for power decay α^d)", 0.01, 0.99, 0.5, 0.01)
    lmbda     = st.sidebar.number_input("lambda (for exp decay)", 0.01, 10.0, 1.0, 0.05)
    DMAX      = st.sidebar.number_input("Max distance (Dmax)", 1, 10, 3, 1)
    TOPK      = st.sidebar.number_input("Sparsify: top-k neighbors per gene", 5, 200, 25, 1)
    RESOLUTION= st.sidebar.number_input("Clustering resolution (Leiden)", 0.1, 5.0, 1.0, 0.1)

    KEEP_ISOLATES = st.sidebar.checkbox("Keep genes with no family membership", value=False)
    COUNT_FILTER  = st.sidebar.slider("Exclude genes with count ≤ N", 0, 90, 0, step=1)
    COUNT_WEIGHTING = st.sidebar.checkbox("Weight similarity by count (importance)", value=False)
    COUNT_GAIN = st.sidebar.slider("Count weight gain (multiplier)", 0.0, 2.0, 0.8, step=0.1)

    SIZE_BY_COUNT = st.sidebar.checkbox("Scale dot size by count (UMAP)", value=True)
    SIZE_MIN, SIZE_MAX = st.sidebar.slider("Dot size range", 1, 28, (1, 18))
    COUNT_SIZE_TH = st.sidebar.slider("Size threshold (count ≥ T scales up)", 1, 50, 5)
    SIZE_BASE = st.sidebar.slider("Size for counts < T", 1, 10, 1)
    SIZE_SCALE_MODE = st.sidebar.selectbox("Size scale above T", ["log","linear"], index=0)

st.sidebar.divider()

st.sidebar.header("Curated file settings")
COUNT_COLUMN_NAME = st.sidebar.text_input(
    "Curated file의 count 열 이름 (예: fold change, count, etc.)",
    value="count"
)
st.sidebar.divider()

RUN = st.sidebar.button("Run pipeline", type="primary")

# Global search (affects UMAP + Text summaries)
GLOBAL_QUERY = st.text_input("🔎 Search gene (symbol or HGNC ID)", value="")
INCLUDE_NONCUR = st.checkbox("Include non-curated neighbors in text summary", value=True)


# -------------------------
# Data loading
# -------------------------
@st.cache_data(show_spinner=False)
def load_library(hgnc_path, clos_path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    hgnc = pd.read_csv(hgnc_path, sep="\t", dtype=str)
    hgnc.columns = [c.strip() for c in hgnc.columns]
    need = {'hgnc_id','symbol','gene_group','gene_group_id'}
    missing = need - set(hgnc.columns)
    if missing:
        raise ValueError(f"hgnc_complete_set.txt missing columns: {missing}")

    clos = pd.read_csv(clos_path, dtype=str)
    clos.columns = [c.strip() for c in clos.columns]
    ren = {}
    if 'parent_fam_id' in clos.columns: ren['parent_fam_id'] = 'ParentGroupID'
    if 'child_fam_id'  in clos.columns: ren['child_fam_id']  = 'ChildGroupID'
    if 'distance'      in clos.columns: ren['distance']      = 'distance'
    clos = clos.rename(columns=ren)
    need2 = {'ParentGroupID','ChildGroupID','distance'}
    if not need2 <= set(clos.columns):
        raise ValueError("hierarchy_closure.csv needs columns: parent_fam_id, child_fam_id, distance")
    clos['distance'] = pd.to_numeric(clos['distance'], errors='coerce').fillna(0).astype(int)
    return hgnc, clos

@st.cache_data(show_spinner=False)
def load_curated(cur_path, count_col_name: str = "count") -> pd.DataFrame:
    cur = pd.read_csv(cur_path, dtype=str)
    cur.columns = [c.strip().lower() for c in cur.columns]
    if 'symbol' not in cur.columns:
        raise ValueError("curated_genes.csv requires a 'symbol' column.")
    count_col_lower = count_col_name.strip().lower()
    if count_col_lower in cur.columns:
        cur['count'] = pd.to_numeric(cur[count_col_lower], errors='coerce').fillna(1).astype(float)
    else:
        st.warning(f"'{count_col_name}' 열을 찾을 수 없어 기본값 1로 처리합니다.")
        cur['count'] = 1.0
    return cur[['symbol','count']]

def attach_groups_for_curated(hgnc: pd.DataFrame, cur: pd.DataFrame,
                              KEEP_ISOLATES=False, COUNT_FILTER=0):
    cur['symbol_up'] = cur['symbol'].str.upper()
    hgnc['symbol_up'] = hgnc['symbol'].str.upper()
    sub = hgnc.merge(cur[['symbol_up','count']], on='symbol_up', how='right')
    sub.rename(columns={'count':'Count'}, inplace=True)

    rows = []
    for _, r in sub.iterrows():
        gid = str(r['hgnc_id'])
        sym = str(r['symbol'])
        cnt = float(r['Count']) if not pd.isna(r['Count']) else 1.0
        gids = parse_group_ids(r['gene_group_id'])
        if not gids:
            if KEEP_ISOLATES:
                rows.append((gid, sym, None, cnt))
            continue
        for gg in gids:
            rows.append((gid, sym, str(gg), cnt))
    df = pd.DataFrame(rows, columns=["GeneID","GeneSymbol","GeneGroupID","Count"])

    unresolved = sorted(set(cur['symbol_up']) - set(sub['symbol_up'].dropna()))
    excluded_by_count = int((cur['count'] <= COUNT_FILTER).sum())
    cur_filtered_syms = set(cur.loc[cur['count'] > COUNT_FILTER, 'symbol_up'])

    df = df[df['GeneSymbol'].str.upper().isin(cur_filtered_syms)]
    return df, unresolved, excluded_by_count


# -------------------------
# Similarity / Clustering / UMAP
# -------------------------
def build_similarity(genes_df: pd.DataFrame, g2g_df: pd.DataFrame, clos: pd.DataFrame,
                     use_exp=False, alpha=0.5, lmbda=1.0, DMAX=3, TOPK=25):
    gene_ids = sorted(genes_df['GeneID'].unique().tolist())
    gid_to_idx = {g:i for i,g in enumerate(gene_ids)}

    groups = sorted(g2g_df['GeneGroupID'].dropna().astype(str).unique().tolist())
    grp_to_idx = {g:i for i,g in enumerate(groups)}

    # Genes x Groups membership with group-size normalization
    gsize = g2g_df.groupby('GeneGroupID')['GeneID'].nunique().to_dict()
    row, col, data = [], [], []
    for _, r in g2g_df.dropna(subset=['GeneGroupID']).iterrows():
        gi = gid_to_idx[str(r['GeneID'])]
        gj = grp_to_idx[str(r['GeneGroupID'])]
        w = 1.0 / math.sqrt(max(1, gsize.get(str(r['GeneGroupID']), 1)))
        row.append(gi); col.append(gj); data.append(w)
    A = sparse.csr_matrix((data, (row, col)), shape=(len(gene_ids), len(groups)), dtype=np.float32)

    # Groups x Groups W with decay, within D<=DMAX
    clos_use = clos[clos['distance'] <= int(DMAX)].copy()
    if use_exp:
        clos_use['w'] = np.exp(-float(lmbda) * clos_use['distance'].astype(float))
    else:
        clos_use['w'] = (float(alpha) ** clos_use['distance'].astype(float))

    row, col, data = [], [], []
    for _, r in clos_use.iterrows():
        p = str(r['ParentGroupID']); c = str(r['ChildGroupID']); w = float(r['w'])
        if p in grp_to_idx and c in grp_to_idx:
            i = grp_to_idx[p]; j = grp_to_idx[c]
            row += [i,j]; col += [j,i]; data += [w,w]
    for g in groups:
        i = grp_to_idx[g]
        row.append(i); col.append(i); data.append(1.0)
    W = sparse.csr_matrix((data, (row, col)), shape=(len(groups), len(groups)), dtype=np.float32)

    S = (A @ W @ A.T).tocsr()
    S.setdiag(0.0)

    # Optional count weighting
    def apply_count_weight(S_sym, gene_ids, genes_df, gain=0.8):
        cmap = dict(genes_df[['GeneID','Count']].values)
        counts = np.array([float(cmap.get(g, 1.0)) for g in gene_ids], dtype=float)
        v = np.log1p(counts)
        lo, hi = np.quantile(v, [0.05, 0.95])
        z = np.zeros_like(v) if (hi-lo)<1e-9 else np.clip((v-lo)/(hi-lo), 0, 1)
        w = 1.0 + gain*z
        D = sparse.diags(w)
        Sw = (D @ S_sym @ D).tocsr()
        Sw.setdiag(0.0)
        return Sw

    if COUNT_WEIGHTING and COUNT_GAIN > 0:
        S = apply_count_weight(S, gene_ids, genes_df, gain=float(COUNT_GAIN))

    # Sparsify: keep top-k per row
    def topk_per_row(M: sparse.csr_matrix, k: int) -> sparse.csr_matrix:
        M = M.tolil()
        for i in range(M.shape[0]):
            row_data = M.data[i]; row_idx = M.rows[i]
            if len(row_idx) > k:
                order = np.argsort(row_data)[::-1][:k]
                M.rows[i] = list(np.array(row_idx)[order])
                M.data[i] = list(np.array(row_data)[order])
        return M.tocsr()

    S = topk_per_row(S, int(TOPK))
    S = S.maximum(S.T).tocsr()
    S.setdiag(0.0)
    return S, gene_ids

def cluster_graph(S_sym: sparse.csr_matrix, RESOLUTION: float) -> np.ndarray:
    coo = S_sym.tocoo()
    edges = list(zip(coo.row.tolist(), coo.col.tolist()))
    weights = coo.data.tolist()
    g = ig.Graph(n=S_sym.shape[0], edges=edges, directed=False)
    g.es['weight'] = weights
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                    weights='weight', resolution_parameter=float(RESOLUTION))
    return np.array(part.membership, dtype=int)

def umap_embedding(S_sym: sparse.csr_matrix, base_neighbors: int = 15) -> np.ndarray:
    """
    Robust UMAP: build dense precomputed distance so every row has >= n_neighbors finite distances.
    """
    n = S_sym.shape[0]
    if n <= 2:
        return np.zeros((n, 2), dtype=np.float32)
    maxS = float(S_sym.max()) if S_sym.nnz > 0 else 1.0
    if maxS <= 0: maxS = 1.0
    D = np.full((n, n), fill_value=maxS, dtype=np.float32)
    coo = S_sym.tocoo()
    D[coo.row, coo.col] = np.maximum(0.0, maxS - coo.data).astype(np.float32)
    np.fill_diagonal(D, 0.0)
    n_neighbors = int(max(2, min(base_neighbors, n - 1)))
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.2, metric='precomputed', random_state=42)
    return reducer.fit_transform(D)


# -------------------------
# Text-only neighbors helpers (FULL HGNC)
# -------------------------
@st.cache_data(show_spinner=False)
def build_full_g2g_from_hgnc(hgnc_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in hgnc_df.dropna(subset=['gene_group_id']).iterrows():
        gid = str(r['hgnc_id'])
        sym = str(r['symbol'])
        for gg in parse_group_ids(r['gene_group_id']):
            rows.append((gid, sym, str(gg)))
    return pd.DataFrame(rows, columns=['GeneID','GeneSymbol','GeneGroupID'])

@st.cache_data(show_spinner=False)
def make_group_distance_map(clos_use_df: pd.DataFrame) -> dict:
    dist = {}
    for _, r in clos_use_df.iterrows():
        p, c, d = str(r['ParentGroupID']), str(r['ChildGroupID']), int(r['distance'])
        if (p,c) not in dist or d < dist[(p,c)]: dist[(p,c)] = d
        if (c,p) not in dist or d < dist[(c,p)]: dist[(c,p)] = d
    return dist


# -------------------------
# Main
# -------------------------
st.title("HGNC Hierarchy-aware Gene Explorer")

if RUN:
    try:
        # Load files
        if UPLOAD_MODE == "Upload files":
            if not (hgnc_file and clos_file and curated_file):
                st.error("Please upload all three files.")
                st.stop()
            hgnc, clos = load_library(hgnc_file, clos_file)
            cur = load_curated(curated_file, COUNT_COLUMN_NAME)
        else:
            if not (Path(hgnc_file).exists() and Path(clos_file).exists() and Path(curated_file).exists()):
                st.error("Path mode: one or more files not found.")
                st.stop()
            hgnc, clos = load_library(str(hgnc_file), str(clos_file))
            cur = load_curated(str(curated_file), COUNT_COLUMN_NAME)

        name_map = build_name_map(hgnc)

        # Curated join (subset gene↔group)
        g2g_df, unresolved, excluded_by_count = attach_groups_for_curated(
            hgnc, cur.copy(), KEEP_ISOLATES=KEEP_ISOLATES, COUNT_FILTER=COUNT_FILTER
        )
        genes_df = g2g_df[['GeneID','GeneSymbol','Count']].dropna(subset=['GeneID']).drop_duplicates().reset_index(drop=True)

        curated_total = int(cur.shape[0])
        dropped = 0
        if not KEEP_ISOLATES:
            curated_all = set(cur['symbol'].str.upper())
            kept_syms = set(genes_df['GeneSymbol'].str.upper())
            dropped = int(max(0, len(curated_all - kept_syms)))

        # Hierarchy cutoff (for S and neighbors)
        clos_use = clos[clos['distance'] <= int(DMAX)].copy()

        # Similarity, clustering, embedding
        with st.spinner("Building similarity & clustering…"):
            S_sym, gene_ids = build_similarity(genes_df, g2g_df, clos_use,
                                               use_exp=use_exp, alpha=alpha, lmbda=lmbda,
                                               DMAX=DMAX, TOPK=int(TOPK))
            labels = cluster_graph(S_sym, float(RESOLUTION))
            XY = umap_embedding(S_sym)

        clusters_df = pd.DataFrame({
            'GeneID': gene_ids,
            'GeneSymbol': [genes_df.set_index('GeneID').get('GeneSymbol', pd.Series()).get(g, "") for g in gene_ids],
            'cluster': labels
        })
        embedding_df = pd.DataFrame({'x': XY[:,0], 'y': XY[:,1]})

        # FULL HGNC maps for non-curated neighbors
        g2g_full_df = build_full_g2g_from_hgnc(hgnc)
        gene2groups_full = defaultdict(set)
        group2genes_full = defaultdict(set)
        for _, r in g2g_full_df.iterrows():
            gene2groups_full[str(r['GeneID'])].add(str(r['GeneGroupID']))
            group2genes_full[str(r['GeneGroupID'])].add(str(r['GeneID']))
        group_dist = make_group_distance_map(clos_use)

        symbol_to_gid = {str(r['GeneSymbol']).upper(): str(r['GeneID']) for _, r in genes_df.iterrows()}
        gid_to_symbol = {v: k for k, v in symbol_to_gid.items()}
        gid2count = {str(r['GeneID']): float(r.get('Count', 0.0)) for _, r in genes_df.iterrows()}
        CURATED_GIDS = set(genes_df['GeneID'].astype(str))
        gid_to_symbol_full = {str(r['hgnc_id']): str(r['symbol'])
                      for _, r in hgnc[['hgnc_id','symbol']].dropna().iterrows()}
        symbol_to_gid_full = {str(r['symbol']).upper(): str(r['hgnc_id'])
                      for _, r in hgnc[['hgnc_id','symbol']].dropna().iterrows()}

        st.session_state['pipeline'] = {
            'hgnc': hgnc,
            'clos_use': clos_use,
            'name_map': name_map,

            'genes_df': genes_df,
            'g2g_df': g2g_df,
            'clusters_df': clusters_df,
            'embedding_df': embedding_df,
            'S_sym': S_sym,
            'labels': np.array(labels),
            'gene_ids': gene_ids,

            'g2g_full_df': g2g_full_df,
            'gene2groups_full': gene2groups_full,
            'group2genes_full': group2genes_full,
            'group_dist': group_dist,

            'symbol_to_gid': symbol_to_gid,
            'gid_to_symbol': gid_to_symbol,
            'gid2count': gid2count,
            'CURATED_GIDS': CURATED_GIDS,

            'gid_to_symbol_full': gid_to_symbol_full,
            'symbol_to_gid_full': symbol_to_gid_full,

            'curated_total': curated_total,
            'excluded_by_count': excluded_by_count,
            'unresolved': unresolved,
            'dropped': dropped,
        }
        st.success("Pipeline complete.")
    except Exception as e:
        st.exception(e)


# -------------------------
# UI after pipeline
# -------------------------
if 'pipeline' in st.session_state:
    P = st.session_state['pipeline']
    genes_df = P['genes_df']; g2g_df = P['g2g_df']
    clusters_df = P['clusters_df']; embedding_df = P['embedding_df']
    S_sym = P['S_sym']; labels = P['labels']; gene_ids = P['gene_ids']
    hgnc = P['hgnc']; clos_use = P['clos_use']; name_map = P['name_map']

    g2g_full_df = P['g2g_full_df']
    gene2groups_full = P['gene2groups_full']; group2genes_full = P['group2genes_full']; group_dist = P['group_dist']
    symbol_to_gid = P['symbol_to_gid']; gid_to_symbol = P['gid_to_symbol']
    gid2count = P['gid2count']; CURATED_GIDS = P['CURATED_GIDS']
    gid_to_symbol_full = P['gid_to_symbol_full']
    symbol_to_gid_full = P['symbol_to_gid_full']


    curated_total = P['curated_total']; excluded_by_count = P['excluded_by_count']
    unresolved = P['unresolved']; dropped = P['dropped']

    # ---- Metrics
    n_clusters = int(np.unique(labels).size)
    avg_csize = int(round(len(gene_ids)/n_clusters)) if n_clusters>0 else 0
    colA, colB, colC, colD, colE, colF, colG = st.columns(7)
    colA.metric("Curated symbols", f"{curated_total}")
    colB.metric("Matched in HGNC", f"{len(genes_df)}")
    colC.metric("Genes w/ groups", f"{g2g_df['GeneID'].nunique()}")
    colD.metric("Dropped (no groups)", f"{dropped}")
    colE.metric("Excluded by filter", f"{excluded_by_count} (≤{COUNT_FILTER})")
    colF.metric("Clusters", f"{n_clusters}")
    colG.metric("Avg cluster size", f"{avg_csize}")

    st.caption(
        f"Parameters: α≈{alpha:.2f} · Dmax={DMAX} · top-k={TOPK} · resolution={RESOLUTION} — "
        "move the Detail level to the right for more/finer clusters."
    )

    # ---- UMAP
    st.subheader("UMAP (genes colored by cluster)")
    umap_df = clusters_df.copy()
    umap_df['x'] = embedding_df['x'].values
    umap_df['y'] = embedding_df['y'].values
    count_map = dict(genes_df[['GeneID','Count']].values)
    umap_df['Count'] = umap_df['GeneID'].map(count_map).fillna(1.0).astype(float)

    def _sizes_from_count(cnt_series, smin, smax, th, base, mode):
        cnt = cnt_series.astype(float).fillna(1.0).to_numpy()
        size = np.full(cnt.shape, float(base), dtype=float)
        mask = cnt >= float(th)
        if mask.any():
            above = cnt[mask]
            vals = (above - float(th)) if mode == 'linear' else np.log1p(above - float(th) + 1.0)
            lo, hi = np.quantile(vals, [0.05, 0.95]) if vals.size > 1 else (vals.min(), vals.max()+1e-9)
            denom = (hi - lo) if (hi - lo) > 1e-9 else 1.0
            t = np.clip((vals - lo) / denom, 0, 1)
            size[mask] = smin + (smax - smin) * t
        return size

    umap_df['size'] = _sizes_from_count(umap_df['Count'], float(SIZE_MIN), float(SIZE_MAX),
                                        float(COUNT_SIZE_TH), float(SIZE_BASE), SIZE_SCALE_MODE)

    nclust = int(np.unique(labels).size)
    colors = (px.colors.qualitative.Plotly * ((nclust // len(px.colors.qualitative.Plotly)) + 1))[:nclust]
    fig_scatter = px.scatter(
        umap_df,
        x='x', y='y',
        color=umap_df['cluster'].astype(str),
        hover_name='GeneSymbol',
        size=('size' if SIZE_BY_COUNT else None),
        color_discrete_sequence=colors,
        height=600,
        hover_data={'Count': True}
    )
    fig_scatter.update_layout(margin=dict(l=10,r=10,t=10,b=10), legend_title_text='cluster')

    # UMAP highlight for global search
    def resolve_search_query_for_umap(q: str) -> Set[str]:
        if not q: return set()
        U = q.strip().upper()
        out = set()
        if U in symbol_to_gid: out.add(symbol_to_gid[U])
        if U.startswith("HGNC:"): out.add(U)
        out |= {gid for sym, gid in symbol_to_gid.items() if U in sym}
        out |= {gid for gid in CURATED_GIDS if U in gid.upper()}
        return out

    matched_gids = resolve_search_query_for_umap(GLOBAL_QUERY)
    msk = umap_df['GeneID'].astype(str).isin(matched_gids)
    if msk.any():
        fig_scatter.add_scatter(
            x=umap_df.loc[msk, 'x'], y=umap_df.loc[msk, 'y'],
            mode='markers+text', name='Search match',
            marker=dict(size=16, line=dict(width=2), color='#e60026', symbol='circle-open'),
            text=umap_df.loc[msk, 'GeneSymbol'], textposition='top center'
        )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # # Cluster size distribution
    # with st.expander("Cluster size distribution (chart + table)", expanded=False):
    #     size_df = clusters_df.groupby('cluster').size().rename('size').reset_index()
    #     size_df['cluster'] = size_df['cluster'].astype(int)
    #     fig_sizes = px.bar(size_df.sort_values('cluster'), x='cluster', y='size', height=280)
    #     st.plotly_chart(fig_sizes, use_container_width=True)
    #     st.dataframe(size_df.sort_values('size', ascending=False).rename(columns={'size':'n_genes'}),
    #                  use_container_width=True, height=300)
    #     st.download_button("Download cluster sizes",
    #                        data=size_df.to_csv(index=False).encode('utf-8'),
    #                        file_name='cluster_sizes.csv')

    # Cluster size distribution
    with st.expander("Cluster size distribution (chart + table)", expanded=False):
        size_df = clusters_df.groupby('cluster').size().rename('size').reset_index()
        size_df['cluster'] = size_df['cluster'].astype(int)
        fig_sizes = px.bar(size_df.sort_values('cluster'), x='cluster', y='size', height=280)
        st.plotly_chart(fig_sizes, use_container_width=True)

        # === 추가: mean fold change per cluster ===
        mean_fc_df = umap_df.groupby('cluster')['Count'].mean().reset_index(name='mean_fold_change')
        fig_mean_fc = px.bar(mean_fc_df.sort_values('cluster'), x='cluster', y='mean_fold_change', height=280,
                            title="Mean fold change per cluster")
        st.plotly_chart(fig_mean_fc, use_container_width=True)

        st.dataframe(size_df.sort_values('size', ascending=False).rename(columns={'size':'n_genes'}),
                    use_container_width=True, height=300)
        st.download_button("Download cluster sizes",
                          data=size_df.to_csv(index=False).encode('utf-8'),
                          file_name='cluster_sizes.csv')

    # Cluster tables
    # with st.expander("Cluster tables (select a cluster to view genes)", expanded=False):
    #     clist_sorted = sorted(map(int, np.unique(labels)))
    #     sel_ctab = st.selectbox("Choose cluster", options=clist_sorted, index=0, key='cluster_table_select')
    #     df_ctab = umap_df[umap_df['cluster']==sel_ctab][['GeneSymbol','Count']].sort_values(['Count','GeneSymbol'], ascending=[False, True]).reset_index(drop=True)
    #     st.dataframe(df_ctab, use_container_width=True, height=300)
    #     st.download_button(
    #         label=f"Download cluster {sel_ctab} gene list",
    #         data=df_ctab.to_csv(index=False).encode('utf-8'),
    #         file_name=f'cluster_{sel_ctab}_genes.csv'
    #     )
    # Cluster tables & Enrichment Analysis
    with st.expander("🔍 Cluster Analysis & Enrichment (GO/KEGG)", expanded=False):
        clist_sorted = sorted(map(int, np.unique(labels)))
        sel_ctab = st.selectbox("Choose cluster", options=clist_sorted, index=0, key='cluster_table_select')
        
        # 데이터 준비
        df_ctab = umap_df[umap_df['cluster']==sel_ctab][['GeneSymbol','GeneID','Count']].sort_values(['Count','GeneSymbol'], ascending=[False, True]).reset_index(drop=True)
        
        # 화면 분할 (왼쪽: 테이블, 오른쪽: 농축 분석 그래프)
        col_tab, col_enr = st.columns([1, 1])
        
        with col_tab:
            st.markdown(f"### Cluster {sel_ctab} Genes")
            st.dataframe(df_ctab, use_container_width=True, height=400)
            st.download_button(
                label=f"Download cluster {sel_ctab} list",
                data=df_ctab.to_csv(index=False).encode('utf-8'),
                file_name=f'cluster_{sel_ctab}_genes.csv'
            )

        with col_enr:
            st.markdown("### Functional Enrichment")
            st.caption("g:Profiler를 사용하여 GO Term 및 KEGG Pathway를 분석합니다.")
            
            if st.button(f"Run Enrichment Analysis", key="btn_enr"):
                # 유전자 심볼 리스트 추출 (공백 제외)
                gene_list = df_ctab['GeneSymbol'].dropna().unique().tolist()
                gene_list = [g for g in gene_list if str(g).strip() != "" and str(g).upper() != "NAN"]
                
                if len(gene_list) < 3:
                    st.warning("분석을 위해 최소 3개 이상의 유전자 심볼이 필요합니다.")
                else:
                    with st.spinner("g:Profiler API 호출 중..."):
                        try:
                            gp = GProfiler(return_pandas=True)
                            # hsapiens(인간), 유의미한 결과만 필터링
                            enr = gp.profile(organism='hsapiens', query=gene_list, 
                                            sources=['GO:BP', 'GO:MF', 'KEGG'])
                            
                            if enr is not None and not enr.empty:
                                # p-value 기준 상위 15개 시각화
                                enr = enr.sort_values('p_value').head(15)
                                enr['-log10(p)'] = -np.log10(enr['p_value'])
                                
                                
                                
                                fig_enr = px.bar(
                                    enr, 
                                    x='-log10(p)', 
                                    y='name', 
                                    color='source',
                                    orientation='h',
                                    title=f"Top Terms for Cluster {sel_ctab}",
                                    labels={'name': 'Biological Term', '-log10(p)': '-log10(P)'},
                                    color_discrete_map={'GO:BP':'#636EFA', 'GO:MF':'#EF553B', 'KEGG':'#00CC96'}
                                )
                                fig_enr.update_layout(yaxis={'categoryorder':'total ascending'}, height=450)
                                st.plotly_chart(fig_enr, use_container_width=True)
                                
                                # 상세 결과 테이블
                                with st.expander("View full enrichment table"):
                                    st.dataframe(enr[['source', 'native', 'name', 'p_value']], use_container_width=True)
                            else:
                                st.info("유의미한(p < 0.05) 분석 결과가 없습니다.")
                        except Exception as e:
                            st.error(f"분석 중 오류 발생: {e}")


    # -------------------------
    # Search results — Text summaries
    # -------------------------
    st.subheader("Search results — text summary")

    def resolve_query_to_gids(q: str) -> set:
        if not q: return set()
        U = q.strip().upper()
        out = set()
        if U in symbol_to_gid: out.add(symbol_to_gid[U])
        if U.startswith("HGNC:"): out.add(U)
        out |= {gid for sym, gid in symbol_to_gid.items() if U in sym}
        out |= {gid for gid in CURATED_GIDS if U in gid.upper()}
        return out

    if GLOBAL_QUERY.strip():
        seeds = resolve_query_to_gids(GLOBAL_QUERY)

        # (A) Same-cluster curated genes
        st.markdown("### (A) Same-cluster curated genes")
        if not seeds:
            st.info("No curated seed matched. (If you searched a non-curated gene, (A) can be empty.)")
        for sg in sorted(seeds):
            if sg in CURATED_GIDS:
                cl = int(clusters_df.loc[clusters_df['GeneID']==sg, 'cluster'].values[0])

                # 1) 해당 클러스터의 기본 표준 컬럼만 추출 (심볼/ID 충돌 회피)
                sub = clusters_df.loc[clusters_df['cluster'] == cl, ['GeneID', 'GeneSymbol']].copy()
                # 2) Count 매핑(genes_df에서 가져옴)
                count_map = dict(genes_df[['GeneID', 'Count']].values)
                sub['Count'] = sub['GeneID'].map(count_map).fillna(1.0)
                # 3) 심볼 폴백: 비어있거나(NaN/공백) 없는 경우 HGNC ID로 대체
                mask_empty = sub['GeneSymbol'].isna() | (sub['GeneSymbol'].astype(str).str.strip() == '')
                sub.loc[mask_empty, 'GeneSymbol'] = sub.loc[mask_empty, 'GeneID']
                # 4) 원하는 컬럼 순서 + 정렬
                sub = (
                    sub[["GeneSymbol", "GeneID", "Count"]]
                    .sort_values(["Count","GeneSymbol"], ascending=[False, True])
                    .reset_index(drop=True)
                    )
                st.caption(f"Seed: {gid_to_symbol.get(sg, '') or '(no symbol)'}  |  HGNC: {sg}  |  Cluster: {cl}  |  n={len(sub)}")
                st.dataframe(sub, use_container_width=True, height=260)
                st.download_button(
                    f"Download cluster {cl} (CSV)",
                    data=sub.to_csv(index=False).encode('utf-8'),
                    file_name=f"cluster_{cl}_genes.csv",
                    mime="text/csv",
                    key=f"dl_cluster_{cl}_{sg}"
                )
            else:
                st.caption(f"Seed {sg}: not curated → no cluster assignment")

        # (B) Nearby genes via family hierarchy (non-curated)
        st.markdown("### (B) Nearby genes via family hierarchy (non-curated)")
        # Seed groups
        seed_groups = set().union(*(gene2groups_full.get(g, set()) for g in seeds))
        # Expand groups within D≤DMAX
        neighbor_groups = set(seed_groups)
        if seed_groups:
            clos_hit = clos_use[clos_use['distance'] <= int(DMAX)]
            mask = clos_hit['ParentGroupID'].astype(str).isin(seed_groups) | clos_hit['ChildGroupID'].astype(str).isin(seed_groups)
            g_hits = set(clos_hit.loc[mask, 'ParentGroupID'].astype(str)) | set(clos_hit.loc[mask, 'ChildGroupID'].astype(str))
            neighbor_groups |= g_hits
        # Genes in those groups (FULL HGNC)
        neighbor_all = set().union(*(group2genes_full.get(gr, set()) for gr in neighbor_groups))
        neighbor_noncur = (neighbor_all - CURATED_GIDS) - set(seeds)
        if not INCLUDE_NONCUR:
            neighbor_noncur = set()

        def min_group_distance_to_seeds(gid: str) -> int:
            gs = gene2groups_full.get(gid, set())
            if not seed_groups or not gs:
                return -1
            best = 10**9
            for a in seed_groups:
                for b in gs:
                    d = group_dist.get((a,b), 10**9)
                    if d < best: best = d
            return (best if best < 10**9 else -1)

        def shared_group_names(gid: str, k=3) -> str:
            gs = gene2groups_full.get(gid, set())
            shared = (gs & neighbor_groups)
            names = [name_map.get(g, g) for g in list(shared)[:k]]
            return ", ".join(names)

        rows = []
        for g in list(neighbor_noncur)[:2000]:
            rows.append({
                "GeneSymbol": gid_to_symbol_full.get(g, ""),
                "GeneID": g,
                "min_group_dist": min_group_distance_to_seeds(g),
                "example_groups": shared_group_names(g, k=3)
            })
        df_noncur = pd.DataFrame(rows).sort_values(["min_group_dist","GeneSymbol"]).reset_index(drop=True)

        st.caption(f"Non-curated neighbors within D≤{DMAX} groups of the seed families — n={len(df_noncur)}")
        st.dataframe(df_noncur, use_container_width=True, height=320)
        st.download_button(
            "Download non-curated neighbors (CSV)",
            data=df_noncur.to_csv(index=False).encode('utf-8'),
            file_name="noncurated_neighbors_hierarchy.csv",
            mime="text/csv"
        )
    else:
        st.info("Type a gene symbol or HGNC ID in the search box to see text results here.")
