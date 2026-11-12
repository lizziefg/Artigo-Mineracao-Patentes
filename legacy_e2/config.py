"""
Configurações do Pipeline de Patentometria em Inteligência Artificial (CPC G06N)
Sprint E2 - TCC CST em Inteligência Artificial - FATEC Rio Claro
Caminhos estritamente relativos e portáteis para qualquer ambiente de execução.
"""
from pathlib import Path

# Diretórios base portáteis (relativos ao diretório do script)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"

# Criação automática dos diretórios estruturais
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUTS_DIR, MODELS_DIR, FIGURES_DIR, TABLES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Parâmetros do Recorte Tecnológico e Temporal
CPC_SUBCLASS = "G06N"  # Computing arrangements based on specific computational models
START_YEAR = 2019
END_YEAR = 2026

# Dicionário de Ruído Formal de Redação Patentária (Patentese Stopwords)
# Termos legais, procedimentais e estruturais que mascaram os tópicos semânticos reais
PATENTESE_STOPWORDS = {
    "claim", "claims", "claimed", "claiming",
    "wherein", "comprising", "comprises", "comprised", "comprise",
    "embodiment", "embodiments", "exemplary", "illustrative",
    "thereof", "therein", "thereto", "thereby", "therefrom", "herein",
    "plurality", "plurality of", "at least one", "one or more",
    "configured to", "configured", "operable to", "adapted to",
    "method comprising", "system comprising", "apparatus comprising",
    "method", "methods", "system", "systems", "apparatus", "device", "devices",
    "present disclosure", "present invention", "invention", "disclosure",
    "aspect", "aspects", "feature", "features", "step", "steps",
    "accordance", "accordance with", "consistent with", "reference to",
    "respective", "respectively", "corresponding", "correspondingly",
    "associated", "associated with", "coupled", "coupled to",
    "predetermined", "predefined", "selected", "particular",
    "module", "unit", "component", "subsystem", "element", "portion",
    "computer", "computing", "processor", "memory", "storage", "database",
    "executable instructions", "computer readable", "storage medium",
    "instructions stored", "executed by", "processor configured"
}

# Configurações do Baseline (LDA)
LDA_CONFIG = {
    "num_topics_range": [5, 8, 10, 12, 15, 20],
    "default_num_topics": 8,
    "max_iter": 50,
    "random_state": 42,
    "max_features": 5000,
    "min_df": 2,
    "max_df": 0.85
}

# Configurações do Modelo Neural (BERTopic)
BERTOPIC_CONFIG = {
    "embedding_model": "all-MiniLM-L6-v2",
    "umap_n_neighbors": 15,
    "umap_n_components": 5,
    "umap_metric": "cosine",
    "hdbscan_min_cluster_size": 10,
    "hdbscan_min_samples": 5,
    "hdbscan_metric": "euclidean",
    "top_n_words": 10,
    "calculate_probabilities": True,
    "nr_topics": "auto"
}
