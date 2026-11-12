"""
Orquestrador Principal do Pipeline de Patentometria Real em IA (Sprint E3).
Calcula estatísticas macro (100% do corpus real) e executa modelagem neural
(BERTopic vs. LDA com paridade de K) sobre amostra estratificada proporcional
determinística (n = 20.000, random_state = 42) para cada fonte e para o harmonizado.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict
import pandas as pd

from bertopic_pipeline import BERTopicPatentPipeline
from collect_real_patents import (
    LensPatentCollector,
    USPTOOpenDataCollector,
    save_source_dataset,
)
from compare_sources import compare_and_harmonize
from config import (
    HARMONIZED_OUTPUTS_DIR,
    LENS_OUTPUTS_DIR,
    PROCESSED_DATA_DIR,
    STRATIFIED_SAMPLE_SIZE,
    USPTO_OUTPUTS_DIR,
)
from evaluate_metrics import TopicEvaluator
from lda_baseline import LDABaselineModel
from preprocess import PatentTextPreprocessor
from visualize_e3 import (
    plot_cluster_distribution_bar,
    plot_country_choropleth_map,
    plot_temporal_evolution_lines,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _build_default_topic_labels(bertopic_summary: pd.DataFrame) -> Dict[int, str]:
    """Cria rótulos iniciais legíveis a partir das 3 principais palavras c-TF-IDF."""
    labels: Dict[int, str] = {}
    for _, row in bertopic_summary.iterrows():
        tid = int(row["Topic"])
        if tid == -1:
            continue
        rep = row.get("Representation", [])
        if isinstance(rep, str):
            rep = [w.strip(" '[]\"") for w in rep.split(",")]
        top3 = ", ".join(rep[:3]) if rep else f"Tópico {tid}"
        labels[tid] = top3
    return labels


def _stratified_sample_by_year(
    df: pd.DataFrame,
    target_n: int = STRATIFIED_SAMPLE_SIZE,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Seleciona amostra estratificada proporcional por ano de publicação
    mantendo exatamente a distribuição temporal real do corpus original.
    """
    if len(df) <= target_n:
        return df.copy().reset_index(drop=True)

    frac = target_n / len(df)
    sampled_parts = []
    for year, grp in df.groupby("publication_year", sort=True):
        n_year = max(1, int(round(len(grp) * frac)))
        n_year = min(n_year, len(grp))
        sampled_parts.append(grp.sample(n=n_year, random_state=random_state))

    df_sample = pd.concat(sampled_parts, ignore_index=True)
    # Ajuste fino para bater exatamente target_n se arredondamento diferir em poucos itens
    if len(df_sample) > target_n:
        df_sample = df_sample.sample(n=target_n, random_state=random_state)
    return df_sample.sort_values(["publication_year", "patent_id"]).reset_index(drop=True)


def execute_modeling_for_dataset(
    df_full: pd.DataFrame,
    dataset_label: str,
    outputs_root: Path,
    sample_size: int = STRATIFIED_SAMPLE_SIZE,
) -> None:
    """
    1) Calcula estatísticas macro sobre 100% do corpus deduplicado real.
    2) Executa pré-processamento, BERTopic (K orgânico) e LDA (mesmo K) na amostra estratificada.
    3) Exporta métricas, tabelas e gráficos 300 DPI.
    """
    if df_full.empty:
        logger.warning(f"[{dataset_label}] Dataset vazio. Etapa de modelagem ignorada.")
        return

    tables_dir = outputs_root / "tables"
    figures_dir = outputs_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # BLOCO 1: ESTATÍSTICAS MACRO SOBRE 100% DOS DADOS REAIS
    # ------------------------------------------------------------------
    logger.info(f"[{dataset_label}] Computando estatísticas macro sobre {len(df_full)} famílias reais...")
    macro_year = (
        df_full["publication_year"]
        .value_counts()
        .sort_index()
        .rename_axis("publication_year")
        .reset_index(name="familias")
    )
    macro_year["participacao_pct"] = (macro_year["familias"] / len(df_full)) * 100.0
    macro_year.to_csv(tables_dir / "evolucao_temporal_macro.csv", index=False, encoding="utf-8-sig")

    macro_assignees = (
        df_full.groupby("assignee")
        .agg(
            patentes=("patent_id", "count"),
            pais_predominante=("assignee_country", lambda s: s.mode().iloc[0] if not s.mode().empty else "XX"),
        )
        .sort_values("patentes", ascending=False)
        .reset_index()
    )
    macro_assignees["participacao_pct"] = (macro_assignees["patentes"] / len(df_full)) * 100.0
    macro_assignees.to_csv(tables_dir / "ranking_depositantes_macro.csv", index=False, encoding="utf-8-sig")

    country_dist = (
        df_full["assignee_country"]
        .value_counts()
        .rename_axis("pais")
        .reset_index(name="patentes")
    )
    country_dist["participacao_pct"] = (country_dist["patentes"] / len(df_full)) * 100.0
    country_dist.to_csv(tables_dir / "distribuicao_paises_macro.csv", index=False, encoding="utf-8-sig")

    plot_country_choropleth_map(df_full, figures_dir)

    # ------------------------------------------------------------------
    # BLOCO 2: FILTRO DE IDIOMA (INGLÊS) + AMOSTRAGEM ESTRATIFICADA + IA
    # ------------------------------------------------------------------
    preprocessor = PatentTextPreprocessor()
    en_mask = df_full["abstract"].apply(preprocessor.is_english_text)
    df_english = df_full[en_mask].copy().reset_index(drop=True)
    logger.info(
        f"[{dataset_label}] Registros com resumo válido em inglês: {len(df_english)} / {len(df_full)} "
        f"({len(df_english)/len(df_full)*100:.1f}%)."
    )

    df_sample = _stratified_sample_by_year(df_english, target_n=sample_size, random_state=42)
    logger.info(
        f"[{dataset_label}] Amostra estratificada gerada: {len(df_sample)} famílias em inglês "
        f"(de {len(df_english)} elegíveis). Pré-processando textos..."
    )

    cleaned_texts = []
    tokens_list = []
    for _, row in df_sample.iterrows():
        full_text = f"{row.get('title', '')}. {row.get('abstract', '')}"
        clean_text, tokens = preprocessor.process_for_models(full_text)
        cleaned_texts.append(clean_text)
        tokens_list.append(tokens)

    df_proc = df_sample.copy()
    df_proc["cleaned_text"] = cleaned_texts
    df_proc["tokens"] = [",".join(t) for t in tokens_list]
    df_proc = df_proc[df_proc["cleaned_text"].apply(lambda x: len(x.split()) >= 5)].reset_index(drop=True)

    corpus_docs = df_proc["cleaned_text"].tolist()
    pub_years = df_proc["publication_year"].astype(int).tolist()

    # 1. Modelo Neural BERTopic (descobre K via HDBSCAN leaf + redução automática)
    emb_cache = PROCESSED_DATA_DIR / f"embeddings_{dataset_label.lower()}.npy"
    bertopic_pipe = BERTopicPatentPipeline(tables_dir=tables_dir, figures_dir=figures_dir)
    topics, _ = bertopic_pipe.fit_transform(
        corpus_docs,
        timestamps=pub_years,
        embeddings_cache_path=emb_cache,
    )
    df_proc["bertopic_topic"] = topics

    bertopic_topics = bertopic_pipe.get_topic_words_dict(top_n=10)
    k_discovered = max(2, len(bertopic_topics))
    outliers_count = int(sum(1 for t in topics if t == -1))
    logger.info(
        f"[{dataset_label}] BERTopic descobriu K={k_discovered} tópicos válidos "
        f"(outliers HDBSCAN={outliers_count}/{len(topics)} = {outliers_count/len(topics)*100:.2f}%)."
    )

    df_bertopic_summary = bertopic_pipe.export_topics_table()
    bertopic_pipe.export_visualizations(corpus_docs, timestamps=pub_years)

    # 2. Baseline LDA (mesmo K descoberto pelo BERTopic para comparação justa)
    lda = LDABaselineModel(n_topics=k_discovered)
    _, lda_topics = lda.fit_transform(corpus_docs)
    lda.export_topics_summary(lda_topics, tables_dir=tables_dir)

    # 3. Métricas Quantitativas (TD & UMass)
    TopicEvaluator.generate_comparison_report(
        docs=corpus_docs,
        lda_topics=lda_topics,
        bertopic_topics=bertopic_topics,
        tables_dir=tables_dir,
    )

    # 4. Tabelas Analíticas dos Clusters na Amostra Estratificada
    valid = df_proc[df_proc["bertopic_topic"] != -1].copy()
    ct_year = pd.crosstab(valid["bertopic_topic"], valid["publication_year"], margins=True, margins_name="Total")
    ct_year.to_csv(tables_dir / "evolucao_temporal_clusters_amostra.csv", encoding="utf-8-sig")

    sample_assignees = (
        valid.groupby("assignee")
        .agg(
            patentes_amostra=("patent_id", "count"),
            pais_predominante=("assignee_country", lambda s: s.mode().iloc[0] if not s.mode().empty else "XX"),
            cluster_predominante=("bertopic_topic", lambda s: int(s.mode().iloc[0]) if not s.mode().empty else -1),
        )
        .sort_values("patentes_amostra", ascending=False)
        .reset_index()
    )
    sample_assignees.to_csv(tables_dir / "ranking_depositantes_amostra_clusters.csv", index=False, encoding="utf-8-sig")

    df_proc.to_csv(PROCESSED_DATA_DIR / f"processado_{dataset_label.lower()}.csv", index=False, encoding="utf-8-sig")

    # Metadados de reprodutibilidade
    meta = {
        "dataset_label": dataset_label,
        "total_familias_reais": int(len(df_full)),
        "familias_elegiveis_ingles": int(len(df_english)),
        "amostra_estratificada_bruta": int(len(df_sample)),
        "amostra_valida_pos_limpeza": int(len(df_proc)),
        "k_topicos_bertopic": int(k_discovered),
        "outliers_hdbscan_n": outliers_count,
        "outliers_hdbscan_pct": round(outliers_count / len(topics) * 100.0, 2),
        "random_state": 42,
    }
    with open(tables_dir / "metadados_execucao.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 5. Figuras Estáticas 300 DPI (Barras + Linhas)
    topic_labels = _build_default_topic_labels(df_bertopic_summary)
    plot_cluster_distribution_bar(df_proc, topic_labels, figures_dir / "fig_clusters_barras.png")
    plot_temporal_evolution_lines(df_proc, topic_labels, figures_dir / "fig_evolucao_temporal_linhas.png")

    logger.info(f"[{dataset_label}] Processamento completo salvo em: {outputs_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline E3 de Patentometria Real (USPTO/BigQuery + The Lens)")
    parser.add_argument("--source", choices=["uspto", "lens", "all"], default="all", help="Fonte a executar")
    parser.add_argument("--use-processed", action="store_true", help="Carrega CSVs já padronizados em data/processed")
    parser.add_argument("--uspto-file", type=str, default=None, help="Arquivo local CSV/TSV/JSON exportado do USPTO/BigQuery")
    parser.add_argument("--lens-file", type=str, default=None, help="Arquivo local CSV/JSON/JSONL exportado do The Lens")
    parser.add_argument("--sample-size", type=int, default=STRATIFIED_SAMPLE_SIZE, help="Tamanho da amostra estratificada")
    args = parser.parse_args()

    df_uspto = pd.DataFrame()
    df_lens = pd.DataFrame()

    if args.source in {"uspto", "all"}:
        proc_uspto = PROCESSED_DATA_DIR / "patentes_uspto_odp.csv"
        if args.use_processed and proc_uspto.exists():
            df_uspto = pd.read_csv(proc_uspto, low_memory=False)
        elif args.uspto_file:
            df_uspto = USPTOOpenDataCollector().load_local_export(Path(args.uspto_file))
            save_source_dataset(df_uspto, "patentes_uspto_odp.csv")
        execute_modeling_for_dataset(df_uspto, "USPTO_ODP", USPTO_OUTPUTS_DIR, sample_size=args.sample_size)

    if args.source in {"lens", "all"}:
        proc_lens = PROCESSED_DATA_DIR / "patentes_the_lens.csv"
        if args.use_processed and proc_lens.exists():
            df_lens = pd.read_csv(proc_lens, low_memory=False)
        elif args.lens_file:
            df_lens = LensPatentCollector().load_local_export(Path(args.lens_file))
            save_source_dataset(df_lens, "patentes_the_lens.csv")
        execute_modeling_for_dataset(df_lens, "THE_LENS", LENS_OUTPUTS_DIR, sample_size=args.sample_size)

    if args.source == "all" and (not df_uspto.empty or not df_lens.empty):
        _, df_harmonized = compare_and_harmonize(df_uspto, df_lens)
        execute_modeling_for_dataset(df_harmonized, "HARMONIZED", HARMONIZED_OUTPUTS_DIR, sample_size=args.sample_size)


if __name__ == "__main__":
    main()
