"""
Configurações do Pipeline de Patentometria Real em Inteligência Artificial (CPC G06N).
Sprint E3 - TCC CST em Inteligência Artificial - FATEC Rio Claro.
Caminhos estritamente relativos e portáteis, sem dados simulados.
"""
import os
from pathlib import Path

# Diretórios base portáteis
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_USPTO_DIR = RAW_DATA_DIR / "uspto_odp"
RAW_LENS_DIR = RAW_DATA_DIR / "the_lens"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = BASE_DIR / "outputs"
USPTO_OUTPUTS_DIR = OUTPUTS_DIR / "uspto_odp"
LENS_OUTPUTS_DIR = OUTPUTS_DIR / "the_lens"
HARMONIZED_OUTPUTS_DIR = OUTPUTS_DIR / "harmonized"

# Criação automática da árvore de diretórios da Sprint E3
for directory in [
    RAW_USPTO_DIR,
    RAW_LENS_DIR,
    PROCESSED_DATA_DIR,
    USPTO_OUTPUTS_DIR / "tables",
    USPTO_OUTPUTS_DIR / "figures",
    LENS_OUTPUTS_DIR / "tables",
    LENS_OUTPUTS_DIR / "figures",
    HARMONIZED_OUTPUTS_DIR / "tables",
    HARMONIZED_OUTPUTS_DIR / "figures",
]:
    directory.mkdir(parents=True, exist_ok=True)

# Parâmetros do Recorte Tecnológico e Temporal
CPC_SUBCLASS = "G06N"  # Computing arrangements based on specific computational models
START_YEAR = 2019
END_YEAR = 2026

# Endpoints Oficiais e Credenciais via Variáveis de Ambiente
API_CONFIG = {
    "uspto_odp_url": "https://api.uspto.gov/api/v1/patent/applications/search",
    "uspto_api_key_env": "USPTO_API_KEY",
    "lens_api_url": "https://api.lens.org/patent/search",
    "lens_api_token_env": "LENS_API_TOKEN",
    "request_timeout": 45,
}


def get_api_credential(env_name: str, cli_value: str | None = None) -> str | None:
    """Obtém credencial da CLI ou de variável de ambiente sem armazenar segredos no código."""
    if cli_value and cli_value.strip():
        return cli_value.strip()
    val = os.getenv(env_name)
    return val.strip() if val and val.strip() else None


# Dicionário Auditável de Ruído Formal de Redação Patentária (Patentese Stopwords)
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
    "instructions stored", "executed by", "processor configured",
    "example", "provide", "provided", "provides", "include", "includes",
    "including", "based", "based on", "first", "second", "third",
    "determine", "determined", "determining", "receive", "received", "receiving",
    "generate", "generated", "generating", "obtain", "obtained", "obtaining",
    "one", "two", "three", "four", "may", "can", "also", "using", "used", "use",
    "via", "within", "without", "among", "between", "least", "more", "most",
    "such", "said", "thus", "various", "different", "multiple", "several",
    "relating", "relates", "related", "according", "disclosed", "described",
    "shown", "illustrated", "perform", "performed", "performing", "process",
    "processed", "processing", "execute", "executed", "executing", "set",
    "value", "values", "input", "output", "information", "data",
}

# Configurações Determinísticas do Baseline (LDA)
LDA_CONFIG = {
    "default_num_topics": 8,
    "max_iter": 50,
    "random_state": 42,
    "max_features": 5000,
    "min_df": 5,
    "max_df": 0.80,
}

# Configurações Determinísticas do Modelo Neural (BERTopic)
STRATIFIED_SAMPLE_SIZE = 20000

BERTOPIC_CONFIG = {
    "embedding_model": "all-MiniLM-L6-v2",
    "encode_batch_size": 128,
    "umap_n_neighbors": 15,
    "umap_n_components": 5,
    "umap_min_dist": 0.0,
    "umap_metric": "cosine",
    "hdbscan_min_cluster_size": 80,
    "hdbscan_min_samples": 10,
    "hdbscan_cluster_selection_method": "leaf",
    "hdbscan_metric": "euclidean",
    "top_n_words": 10,
    "calculate_probabilities": False,
    "nr_topics": 10,
    "random_state": 42,
    "outlier_reduction_strategy": "c-tf-idf",
    "outlier_reduction_threshold": 0.0,
}
