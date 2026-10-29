# GeneExplorer

GeneExplorer organizes experimental genes with HGNC gene groups and hierarchy. It separates annotation-based seed exploration, clustering, expression-informed ranking, and external assessment.

## Run

Use Python **3.12** and the pinned environment:

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app_0303.py
```

`streamlit_app.py` is an equivalent entry point. Do not rename either entry point to `streamlit.py`, which would shadow the installed Streamlit package. Streamlit Community Cloud should use Python 3.12 and `streamlit_app_0303.py`. The existing deployment filename is preserved.

## Included applications

- **GSE74000, APAP-associated acute liver failure:** the original laboratory-supplied 3,713-symbol list maps exactly to the frozen 2,523-gene comparison. Expression / Family-only / Hierarchy yield 11 / 486 / 397 clusters with the reference settings. Fold change is used only for display; the legacy `count` column is a control-expression mean, not read count. GSE120652 (three cases and three controls) supplies independent external expression assessment.
- **GSE241735, FGFR-inhibited cholangiocarcinoma:** 8,861 genes; ICC13-7 discovery statistics generate rankings and MG69/MG212 PDX response statistics are used for evaluation. The shared gene universe was filtered across all three contrasts. This is explicitly a conditional application case, not completely untouched external validation. Primary top-50 down-response hits for Hierarchy are 16 / 13 / 10 for MG69 / MG212 / Both PDX.
- **Custom upload:** UTF-8 CSV/TSV with unique headers and a symbol/HGNC-ID column. Select individual, already normalized log2/logCPM expression sample columns for an expression baseline (at least three). Raw counts are not normalized or tested inside the app. Optional discovery scores require `hgnc_id,signed_stat`, where signed_stat is sign(log2FC) × sqrt(edgeR quasi-likelihood F). Unambiguous historical aliases can be enabled explicitly. Mapping exclusions are exportable; duplicate mapped IDs and invalid numbers are rejected.

## What the app computes

For family f, membership weight is 1/sqrt(n(f)), using approved genes in the full frozen reference. Family-only uses A Aᵀ; Hierarchy uses A K Aᵀ, with diagonal K=1 and off-diagonal alpha^distance up to the cutoff. Positive top-k neighbors use canonical HGNC-ID tie-breaking and max symmetrization. Reference settings are alpha=0.5, distance=3, k=25 for liver clustering and k=50 for RNA-seq ranking. Liver clustering uses Leiden RB configuration, resolution=1, seed=42 and two iterations, retaining isolates as singletons.

The expression baseline uses positive Pearson correlation of the supplied transformed sample profiles. RNA-seq scores combine unit-norm discovery evidence and one-step row-normalized graph-neighbor evidence with lambda=0.5. Direct DE and degree-only controls are provided. Frozen random/permutation comparisons are displayed only for reference settings; new settings do not inherit their P values.

Seed exploration is a **separate** unsparsified A K Aᵀ binary-seed operation, with same-family and hierarchy contributions. It is not expression propagation, a probability or a validated therapeutic recommendation.

Optional UMAP uses active rows of the sparse Hierarchy affinity graph, cosine metric, 15 neighbors (bounded for small inputs), min_dist=0.1, random initialization, random_state=42 and one job. Isolates remain in cluster outputs and are omitted from the layout. Numeric marker sizes use absolute display values only. This new visualization is exploratory and is not a reproduction of historical screenshot coordinates.

## Interpretation and limits

Hierarchy does not consistently outperform the main baselines. The previous 33.8-fold enrichment ratio is withdrawn. Added-edge external support is exploratory (two-test Holm P=0.04398); the previously isolated-gene endpoint is not significant (P=0.09795). RNA-seq comparisons with Expression, Family-only and Direct DE are not significant after correction. Drug-associated expression responses are not therapeutic-target validation.

Live GO/KEGG analysis is user-triggered. It sends selected gene symbols and the full analyzed gene set as background to g:Profiler, requests FDR correction and retains all returned terms plus service metadata. Separate queries do not receive an across-cluster correction. This interpretation feature is not independent evidence of model performance. Database versions are recorded from the live response, not silently treated as frozen manuscript annotations.

## Reproducibility

Download the complete run ZIP for inputs, graphs, assignments/rankings, optional layout, mapping audit, settings and any enrichment responses. Raw-count edgeR preprocessing remains in the coauthor's offline analysis project; this app uses its frozen prepared tables. The app does not silently change the common RNA-seq gene universe to a new discovery-only experiment.

Source provenance and original checksums are recorded in `docs/SOURCE_PROVENANCE.json`. `legacy/streamlit_app_0303_20260414.py` and `streamlit_app_backupfile.py` are historical code, not current entry points. `_frozen_core.py` is retained from the received project; its old raw-intensity expression function is not called by the app. The corrected, already-transformed baseline adapter is in `geneexplorer/analysis.py`.

```bash
python -m unittest discover -s tests -v
```

Tests compare every frozen liver graph/cluster label, all 8,861 RNA-seq rankings, external metrics, upload validation, family names, seed explanations, exports and Streamlit interactions. They do not claim a fresh raw-data edgeR reanalysis or broad biological superiority. Python dependencies are installed during setup, never automatically while the app runs.

## Publication package

The repository files run the application. The accompanying offline reproduction
and figures/supplement ZIP files are intended as assets of the same GitHub Release.
See [offline reproduction instructions](docs/OFFLINE_REPRODUCTION_EN.txt),
[data provenance](docs/DATA_PROVENANCE_EN.txt), and
[the later real interface run](docs/LATER_INTERFACE_RUN_EN.txt).
Original-input sample roles and numerical provenance are included in `data/`.
The tested computational code is unchanged from local commit
`849c7495a9d5a1bc2e6101a1721d30b19d0922a4`; packaging adds documentation and provenance.
Uploading these files creates a new public commit, which should be recorded separately.
