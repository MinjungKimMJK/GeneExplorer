# HGNC Hierarchy-aware Gene Explorer

### **Overview**
This platform is a specialized bioinformatics tool designed for **gene clustering and functional discovery** by integrating genomic hierarchy with experimental data. Developed particularly for research in **Drug Delivery Systems (DDS)** and **Regulatory Science**, it helps researchers identify latent gene relationships and validate biological mechanisms.

### **Key Features**
* **Hierarchy-aware Clustering**: Utilizes HGNC gene group structures and the **Leiden algorithm** to cluster genes based on functional lineages rather than simple expression proximity.
* **Hyperdiffusion-based Recommendation**: Implements a network diffusion algorithm (Hyperdiffusion) to recommend potential gene candidates (including non-curated genes) related to specific seed clusters or genes.
* **Functional Enrichment Integration**: Direct integration with the **g:Profiler API** for real-time **Gene Ontology (GO)** and **KEGG Pathway** analysis of identified clusters.
* **Interactive Visualization**: High-dimensional data exploration via **UMAP** with dynamic scaling based on user-provided metrics (e.g., Fold Change).

### **Installation & Requirements**
Ensure you have Python 3.8+ installed. It is recommended to use an **Anaconda** environment to prevent library conflicts.

**Required Dependencies:**
The following libraries must be installed (listed in `requirements.txt`):
* `streamlit`, `plotly`
* `pandas`, `numpy`, `scipy`
* `python-igraph`, `leidenalg`
* `umap-learn`
* `gprofiler-official`

### **Usage**
1.  **Launch the application**:
    ```bash
    streamlit run streamlit_app.py
    ```
2.  **Upload Data**:
    * `hgnc_complete_set.txt` (HGNC Dataset)
    * `hierarchy_closure.csv` (Hierarchy Mapping)
    * `Experimental_Data.csv` (Curated gene list with counts/fold-change)
3.  **Analysis**: Adjust the **Detail Level** to refine clusters and perform **Enrichment Analysis** to interpret biological significance.

### **Academic Significance**
This tool supports the development of **regulatory policies for emerging bio-technologies** by providing a transparent and reproducible framework for evaluating gene-level responses to new drug delivery platforms.