"""Streamlit interface for reproducible HGNC exploration and case studies."""
from dataclasses import asdict
import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from . import __version__
from .analysis import (ROOT, Settings, SLUG, digest, read_table, load_example, load_custom,
                       compute, external_coherence, response_metrics, seed_neighbors, pair_explanations, export_run)
from .enrichment import profile_genes


def csv_download(label, frame, filename, key):
    st.download_button(label, frame.to_csv(index=False).encode('utf-8-sig'), filename, 'text/csv', key=key)


def reference_setting(result):
    s = result['settings']
    return (result['dataset'] in ['GSE74000', 'GSE241735'] and s['alpha'] == .5 and s['max_distance'] == 3
            and s['top_k'] == (25 if result['dataset'] == 'GSE74000' else 50)
            and s['resolution'] == 1 and s['mixture'] == .5)


def controls():
    st.sidebar.header('Data')
    dataset = st.sidebar.selectbox('Analysis dataset', ['GSE74000', 'GSE241735', 'Custom'],
                                  format_func=lambda x: {'GSE74000': 'APAP liver — GSE74000',
                                                         'GSE241735': 'FGFR RNA-seq — GSE241735', 'Custom': 'Upload your gene list'}[x], key='dataset')
    custom, valid = {}, True
    if dataset == 'Custom':
        st.info('Upload a UTF-8 CSV/TSV with a symbol or HGNC ID column. Optional expression columns must contain normalized log2/logCPM sample values, not raw counts, group means or fold changes.')
        file = st.file_uploader('Gene list and optional expression samples', type=['csv', 'tsv', 'txt'], key='genes_upload')
        if file is None:
            valid = False
        else:
            try:
                frame = read_table(file.getvalue())
                candidate = next((c for c in frame if c.lower() in ['hgnc_id', 'symbol', 'genesymbol']), frame.columns[0])
                identifier = st.selectbox('Gene identifier column', frame.columns.tolist(), index=list(frame.columns).index(candidate))
                available = [c for c in frame if c != identifier]
                samples = st.multiselect('Normalized expression sample columns (optional)', available)
                display = st.selectbox('Numeric display column (optional)', ['None']+available)
                aliases = st.checkbox('Allow unambiguous aliases and previous symbols', value=False)
                st.dataframe(frame.head(5), hide_index=True, width="stretch")
                custom = dict(payload=file.getvalue(), identifier=identifier, samples=samples,
                              display_column=None if display == 'None' else display, allow_aliases=aliases)
                score = st.file_uploader('Optional discovery statistics for ranking: hgnc_id, signed_stat', type=['csv', 'tsv'], key='score_upload',
                                        help='signed_stat = sign(log2FC) × sqrt(edgeR quasi-likelihood F), calculated from discovery samples only. Raw-count DE is performed in the offline R project.')
                if score is not None: custom['score_payload'] = score.getvalue()
                with st.expander('Use a different HGNC reference'):
                    replace = st.checkbox('Upload both reference files', value=False)
                    if replace:
                        hgnc = st.file_uploader('HGNC complete-set TSV', type=['txt', 'tsv'], key='hgnc_upload')
                        closure = st.file_uploader('Hierarchy closure CSV', type=['csv'], key='closure_upload')
                        if hgnc is None or closure is None:
                            st.warning('Both reference files are required.'); valid = False
                        else: custom.update(hgnc_bytes=hgnc.getvalue(), closure_bytes=closure.getvalue())
            except (ValueError, KeyError, pd.errors.ParserError) as exc:
                st.error(str(exc)); valid = False
        st.download_button('Download input template', 'symbol,sample_1,sample_2,sample_3,display_value\nSAMM50,5.1,5.3,5.2,1.2\nMTX2,4.8,5.0,4.9,1.1\nMTX3,4.5,4.7,4.6,1.3\n',
                           'gene_input_template.csv', 'text/csv', key='template')
    mode = 'ranking' if dataset == 'GSE241735' or 'score_payload' in custom else 'clustering'
    st.sidebar.header('Settings')
    st.sidebar.caption('Reference settings are shown by default. Changes create a new exploratory run.')
    alpha = st.sidebar.slider('Hierarchy decay α', 0., 1., .5, .05, key='alpha_'+dataset)
    distance = st.sidebar.selectbox('Maximum hierarchy distance', [1, 2, 3], index=2, key='distance_'+dataset)
    top_k = st.sidebar.number_input('Positive neighbors per gene', 1, 200, 50 if mode == 'ranking' else 25, 1, key='topk_'+dataset+'_'+mode)
    resolution = st.sidebar.number_input('Leiden resolution', .1, 5., 1., .1, disabled=mode == 'ranking', key='resolution_'+dataset)
    mixture = st.sidebar.slider('Discovery / neighbor mixture λ', 0., 1., .5, .05, disabled=mode != 'ranking', key='mixture_'+dataset,
                                help='0 uses discovery evidence alone; 1 uses neighbor evidence alone.')
    layout = st.sidebar.checkbox('Compute UMAP layout', value=False, disabled=mode == 'ranking', key='layout_'+dataset,
                                 help='Optional exploratory display. It does not affect graphs, clusters or rankings. Isolates stay in the analysis but are omitted from UMAP.')
    settings = Settings(alpha=alpha, max_distance=distance, top_k=int(top_k), resolution=resolution,
                        mixture=mixture, make_layout=layout and mode == 'clustering')
    signature_data = {'dataset': dataset, 'settings': asdict(settings),
                      'custom': {k: digest(v) if isinstance(v, bytes) else v for k, v in custom.items()}}
    signature = digest(json.dumps(signature_data, sort_keys=True).encode())
    clicked = st.sidebar.button('Run analysis', type='primary', key='run_analysis', disabled=not valid)
    st.sidebar.caption('Leiden: seed 42 · two iterations. HGNC family sizes are taken from the complete reference.')
    return dataset, settings, custom, signature, clicked


def show_clusters(result):
    method = st.selectbox('Graph to explore', list(result['graphs']), index=list(result['graphs']).index('Hierarchy'), key='view_method')
    frame = result['genes'].copy().assign(cluster=result['labels'][method])
    symbols = dict(zip(frame.hgnc_id, frame.symbol))
    selected_gene = st.selectbox('Find a gene', ['']+frame.hgnc_id.tolist(),
                                 format_func=lambda g: 'Type a gene symbol or HGNC ID' if not g else f'{symbols[g]} ({g})', key='gene_search')
    if method in result['embeddings']:
        positions = frame.merge(result['embeddings'][method], on='hgnc_id', validate='one_to_one')
        visible = positions.dropna(subset=['x', 'y']).copy()
        if not visible.empty:
            visible['cluster_label'] = visible.cluster.astype(str)
            size = None
            if result['display_label'] and st.checkbox('Scale marker size by absolute display value', value=False):
                visible['marker_size'] = np.log1p(np.abs(visible.display_value.fillna(0)))+.1; size = 'marker_size'
            fig = px.scatter(visible, x='x', y='y', color='cluster_label', hover_name='symbol',
                             hover_data=['hgnc_id', 'display_value'], size=size, height=520)
            matched = visible[visible.hgnc_id == selected_gene]
            if not matched.empty:
                fig.add_scatter(x=matched.x, y=matched.y, text=matched.symbol, mode='markers+text',
                                marker=dict(size=17, symbol='circle-open', color='black'), name='Selected gene')
            st.plotly_chart(fig, use_container_width=True)
        st.caption(f"{result['embedding_metadata']['algorithm']}; graph affinity profiles; cosine metric; random state 42. Isolated genes remain in the cluster table and have no plotted coordinates.")
    else:
        st.caption('UMAP is optional and currently available for the Hierarchy graph. Enable it before running if a display is needed.')
    sizes = frame.groupby('cluster').size().rename('genes').reset_index()
    st.plotly_chart(px.bar(sizes, x='cluster', y='genes', title='Cluster sizes'), use_container_width=True)
    if result['display_label'] and frame.display_value.notna().any():
        means = frame.groupby('cluster').display_value.mean().reset_index()
        st.plotly_chart(px.bar(means, x='cluster', y='display_value', title='Mean '+result['display_label']+' by cluster'), use_container_width=True)
        st.caption('The numeric display column does not change graph weights, cluster assignments or enrichment tests.')
    if selected_gene:
        cl = int(frame.loc[frame.hgnc_id == selected_gene, 'cluster'].iloc[0])
        st.info(f'{symbols[selected_gene]} belongs to cluster {cl}.')
    cluster_id = st.selectbox('Cluster for inspection', sorted(frame.cluster.unique().tolist()), key='cluster_choice')
    members = frame[frame.cluster == cluster_id]
    st.dataframe(members.drop(columns=['input_position']), hide_index=True, width="stretch")
    csv_download('Download selected cluster', members, f'cluster_{cluster_id}.csv', 'dl_cluster')
    csv_download('Download all cluster assignments', frame.drop(columns=['input_position']), SLUG[method]+'_clusters.csv', 'dl_all_clusters')
    enrichment_panel(result, members.symbol.tolist(), f'{SLUG[method]}_cluster_{cluster_id}')


def show_rankings(result):
    st.caption('Scores combine normalized discovery signed √F with one-step graph-neighbor evidence. Positive and negative rankings are evaluated separately.')
    methods = [m for m, slug in SLUG.items() if slug+'_score' in result['rankings']]
    method = st.selectbox('Ranking method', methods, index=methods.index('Hierarchy'), key='ranking_method')
    direction = st.radio('Response direction', ['down', 'up'], horizontal=True, key='rank_direction')
    n = st.selectbox('Show top genes', [10, 25, 50, 100], index=2)
    stem = SLUG[method]
    columns = ['hgnc_id', 'symbol', stem+'_score', stem+'_'+direction+'_rank']
    table = result['rankings'].sort_values(columns[-1])[columns].head(n)
    st.dataframe(table, hide_index=True, width="stretch")
    csv_download('Download all scores and ranks', result['rankings'], 'all_rankings.csv', 'dl_rankings')
    enrichment_panel(result, table.symbol.tolist(), f'{stem}_{direction}_top{n}')


def enrichment_panel(result, query, key):
    with st.expander('Functional interpretation — GO / KEGG'):
        st.caption('Live g:Profiler analysis sends the selected gene symbols and the full analyzed gene set as background. FDR is requested; no correction is applied across separate cluster queries. This is functional annotation, not independent performance evidence.')
        sources = st.multiselect('Annotation sources', ['GO:BP', 'GO:MF', 'GO:CC', 'KEGG'], default=['GO:BP', 'GO:MF', 'KEGG'], key='enrichment_sources')
        request_key = key+'_'+digest('|'.join(sorted(sources)).encode())[:8]
        if st.button('Analyze selected gene set', key='run_enrichment', disabled=len(query) < 3 or not sources):
            try:
                with st.spinner('Requesting annotation results…'):
                    result['enrichments'][request_key] = profile_genes(query, result['genes'].symbol.tolist(), sources)
            except Exception as exc:
                st.error('Enrichment request failed: '+str(exc))
        if request_key in result['enrichments']:
            response = result['enrichments'][request_key]
            full = pd.DataFrame(response['result'])
            if full.empty:
                st.info('No terms were returned for this query and background.')
            else:
                significant = full[full.significant].sort_values('p_value').head(15).copy()
                if significant.empty:
                    st.info('No terms passed the service-adjusted 0.05 threshold. All returned terms remain downloadable.')
                else:
                    significant['minus_log10_adjusted_p'] = -np.log10(significant.p_value.clip(lower=np.finfo(float).tiny))
                    fig = px.bar(significant, x='minus_log10_adjusted_p', y='name', color='source', orientation='h', height=470)
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                csv_download('Download every returned term', full, 'enrichment_all_terms.csv', 'dl_enrichment')
            st.download_button('Download query and database metadata', json.dumps(response, indent=2), 'enrichment_with_metadata.json', 'application/json', key='dl_enrich_meta')


def show_seed_exploration(result):
    st.caption('Annotation-based seed exploration is separate from the RNA-seq response ranking. Its score is unsparsified A K Aᵀ applied to binary seeds; it is not a probability or therapeutic validation.')
    symbols = {g: d['symbol'] for g, d in result['reference'].genes.items() if d['families']}
    seeds = st.multiselect('Seed genes', sorted(symbols), format_func=lambda g: f'{symbols[g]} ({g})', max_selections=10, key='seeds')
    outside = st.checkbox('Show only genes outside the input list', value=True, key='outside_seeds')
    if seeds:
        table = seed_neighbors(result, seeds, outside)
        st.caption(f'{len(table):,} candidates with positive annotation scores; shown in deterministic score order.')
        st.dataframe(table.head(200), hide_index=True, width="stretch")
        csv_download('Download all seed candidates', table, 'seed_candidates.csv', 'dl_seed_candidates')
        if not table.empty:
            candidate = st.selectbox('Explain candidate score', table.hgnc_id.tolist(), format_func=lambda g: f'{symbols[g]} ({g})')
            seed = st.selectbox('Explain relative to seed', seeds, format_func=lambda g: symbols[g])
            explanation = pair_explanations(result, candidate, seed)
            st.dataframe(explanation, hide_index=True, width="stretch")
            csv_download('Download family contributions', explanation, 'family_contributions.csv', 'dl_explanations')


def show_evaluation(result):
    if result['dataset'] == 'GSE74000':
        st.write('Independent liver expression assessment: GSE120652, three cases and three controls.')
        values = external_coherence(result)
        st.dataframe(values, hide_index=True, width="stretch")
        st.caption('Clusters need at least four externally represented genes. Macro correlation weights eligible clusters equally; pair-weighted correlation weights their gene pairs. Six samples provide limited external evidence.')
        csv_download('Download external coherence', values, 'external_coherence.csv', 'dl_external')
        if reference_setting(result):
            tests = pd.read_csv(ROOT/'data/evaluation/GSE74000_paired_primary_comparisons.csv')
            st.write('Frozen full-profile comparisons — 2,000 size-preserving permutations')
            st.dataframe(tests, hide_index=True, width="stretch")
            st.info('Hierarchy is not significantly better than Family-only on centered macro coherence (Holm P=0.72764); Expression correlation is higher than Hierarchy (Holm P=0.01349).')
            mech = pd.read_csv(ROOT/'data/evaluation/gse74000_mechanism_summary.csv')
            mech['Holm_p_two_tests'] = [0.04397801099450275, 0.09795102448775612]
            st.dataframe(mech, hide_index=True, width="stretch")
            st.caption('These are saved coauthor full-profile tests for the reference configuration, not fresh permutation tests by this browser session. Added-edge evidence is exploratory, not overall superiority.')
            with st.expander('SAMM50 and STX17 — post hoc examples'):
                paths = pd.read_csv(ROOT/'data/evaluation/GSE74000_added_edges_with_paths.csv')
                paths = paths[paths.gene_a_symbol.isin(['SAMM50', 'STX17']) | paths.gene_b_symbol.isin(['SAMM50', 'STX17'])]
                cols = ['gene_a_symbol', 'gene_b_symbol', 'supporting_family_path_names', 'external_centered_correlation']
                st.dataframe(paths[cols], hide_index=True, width="stretch")
                st.caption('HGNC paths supply functional context. Correlations are descriptive, not individually significant edges or physical-interaction evidence.')
        else:
            st.info('Settings differ from the reference analysis. The metrics above were recomputed; frozen reference P values are not applied to this run.')
    elif result['dataset'] == 'GSE241735':
        st.write('PDX response evaluation: MG69 and MG212, three treated and three vehicle samples per model.')
        st.info(result['note'])
        metrics = response_metrics(result)
        st.dataframe(metrics, hide_index=True, width="stretch")
        csv_download('Download response metrics', metrics, 'response_metrics.csv', 'dl_response')
        st.caption('Response: FDR ≤0.05 and |log2FC| ≥1 in the matching direction. Both PDX requires both models. These are expression-response endpoints, not validated therapeutic targets.')
        if reference_setting(result):
            comparisons = pd.read_csv(ROOT/'data/evaluation/GSE241735_comparison_to_hierarchy.csv')
            st.write('Frozen top-50 comparisons — Holm adjustment across 30 tests')
            st.dataframe(comparisons, hide_index=True, width="stretch")
            st.info('All 18 comparisons with Expression correlation, Family-only and Direct DE have Holm P=1.0. Numerically higher hit counts do not establish superiority.')
            st.caption('Random comparisons use the saved 2,000-ranking distribution. Tests apply to the frozen settings only; average-precision differences are descriptive.')
        else:
            st.info('The ranking metrics were recomputed for these settings. Reference significance tests are not reused.')
    else:
        st.info('No independent evaluation dataset is attached to this upload. Clusters, seed scores and enrichment do not validate biological or therapeutic effects.')


def main():
    st.set_page_config(page_title='GeneExplorer — HGNC exploration', layout='wide')
    st.title('GeneExplorer')
    st.caption('Explore experimental genes through HGNC groups and hierarchy, with expression evidence where available.')
    dataset, settings, custom, signature, clicked = controls()
    if clicked:
        st.session_state.pop('analysis', None)
        st.session_state.pop('analysis_signature', None)
        for key in ['view_method', 'gene_search', 'cluster_choice', 'ranking_method', 'seeds']:
            st.session_state.pop(key, None)
        try:
            with st.spinner('Mapping genes and building the frozen-model comparisons…'):
                bundle = load_custom(**custom) if dataset == 'Custom' else load_example(dataset)
                result = compute(bundle, settings)
            st.session_state['analysis'] = result
            st.session_state['analysis_signature'] = signature
            st.success('Analysis complete.')
        except (ValueError, KeyError, OSError) as exc:
            st.error('Analysis could not run: '+str(exc))
    if 'analysis' not in st.session_state:
        st.write('Choose an example or upload a gene list, then run the analysis.')
        st.caption('The APAP example uses the corrected 2,523-gene configuration. The RNA-seq example reproduces the received common-universe case; it does not implement a new discovery-only filtering experiment.')
        return
    if st.session_state.get('analysis_signature') != signature:
        st.warning('Inputs or settings have changed. Run the analysis again to display results for this configuration.')
        return
    result = st.session_state['analysis']
    st.caption(result['note'])
    columns = st.columns(3)
    columns[0].metric('Analyzed genes', len(result['genes']))
    columns[1].metric('Input rows', len(result['mapping']))
    columns[2].metric('Excluded rows', int((~result['mapping'].included).sum()))
    st.dataframe(result['structure'], hide_index=True, width="stretch")
    st.caption(f"Run {result['run_id']} · α={settings.alpha:g}, distance={settings.max_distance}, k={settings.top_k}, resolution={settings.resolution:g}, λ={settings.mixture:g}")
    panels = st.tabs(['Rankings' if result['mode'] == 'ranking' else 'Clusters and expression', 'Seed exploration', 'External assessment', 'Data and reproducibility'])
    with panels[0]:
        show_rankings(result) if result['mode'] == 'ranking' else show_clusters(result)
    with panels[1]: show_seed_exploration(result)
    with panels[2]: show_evaluation(result)
    with panels[3]:
        st.write('Mapping and exclusions')
        st.dataframe(result['mapping'], hide_index=True, width="stretch")
        csv_download('Download mapping audit', result['mapping'], 'mapping_audit.csv', 'dl_mapping')
        st.write('Recorded configuration')
        st.json({'app_version': __version__, 'settings': result['settings'], 'input_sha256': result['source_hashes'],
                 'embedding': result.get('embedding_metadata'), 'constant_expression_rows': result['constant_expression_rows']})
        st.download_button('Download complete run ZIP', export_run(result), f"GeneExplorer_{result['run_id']}.zip", 'application/zip', key='dl_run')
        st.caption('Includes inputs, graphs, all assignments or rankings, optional layout coordinates, mapping audit and any enrichment metadata. No historical 33.8-fold claim is used.')
