"""
Orquestrador Principal do Pipeline de Patentometria em IA (Sprint E2)
Executa de ponta a ponta: Ingestão -> Pré-processamento -> LDA -> BERTopic -> Métricas e Gráficos.
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUTS_DIR, TABLES_DIR
from collect_patents import PatentCollector
from preprocess import PatentTextPreprocessor
from lda_baseline import LDABaselineModel
from bertopic_pipeline import BERTopicPatentPipeline
from evaluate_metrics import TopicEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_patent_pipeline(input_file: str = None, use_sample: bool = True, sample_size: int = 500):
    logger.info("=" * 70)
    logger.info("INICIANDO PIPELINE DE MINERAÇÃO PATENTÁRIA EM IA (CPC G06N) - SPRINT E2")
    logger.info("=" * 70)

    # 1. COLETAR / CARREGAR DADOS BRUTOS
    collector = PatentCollector()
    
    if input_file and Path(input_file).exists():
        logger.info(f"Carregando base de patentes fornecida: {input_file}")
        df_raw = collector.load_lens_export(Path(input_file))
    else:
        benchmark_file = RAW_DATA_DIR / "corpus_patentes_cpc_g06n_benchmark.csv"
        if benchmark_file.exists() and not use_sample:
            logger.info(f"Carregando corpus benchmark existente: {benchmark_file.name}")
            df_raw = pd.read_csv(benchmark_file)
        else:
            logger.info(f"Gerando corpus benchmark calibrado para teste imediato ({sample_size} registros)...")
            df_raw = collector.generate_benchmark_corpus(n_samples=sample_size)

    logger.info(f"Total de patentes carregadas para análise: {len(df_raw)}")

    # 2. PRÉ-PROCESSAMENTO ESPECIALIZADO (PATENTESE STOPWORDS)
    logger.info("Iniciando pré-processamento textual e remoção de patentese...")
    preprocessor = PatentTextPreprocessor()
    
    cleaned_texts = []
    tokens_list = []
    
    for idx, row in df_raw.iterrows():
        full_text = f"{row.get('title', '')}. {row.get('abstract', '')}"
        clean_text, tokens = preprocessor.process_for_models(full_text)
        cleaned_texts.append(clean_text)
        tokens_list.append(tokens)

    df_raw["cleaned_text"] = cleaned_texts
    df_raw["tokens"] = [",".join(t) for t in tokens_list]
    
    valid_mask = df_raw["cleaned_text"].apply(lambda x: len(x.split()) >= 5)
    df_processed = df_raw[valid_mask].copy().reset_index(drop=True)
    
    processed_file = PROCESSED_DATA_DIR / "patentes_g06n_limpas.csv"
    df_processed.to_csv(processed_file, index=False, encoding="utf-8")
    logger.info(f"Base pré-processada salva em: {processed_file.name} ({len(df_processed)} documentos válidos)")

    corpus_docs = df_processed["cleaned_text"].tolist()
    pub_years = df_processed["publication_year"].tolist() if "publication_year" in df_processed.columns else None

    # 3. BASELINE: LATENT DIRICHLET ALLOCATION (LDA)
    logger.info("-" * 50)
    logger.info("ETAPA 3: MODELAGEM BASELINE COM LDA")
    logger.info("-" * 50)
    lda = LDABaselineModel(n_topics=8)
    _, lda_topics = lda.fit_transform(corpus_docs)
    df_lda_summary = lda.export_topics_summary(lda_topics)
    print("\n[TÓPICOS EXTRAÍDOS PELO LDA - BASELINE]:")
    print(df_lda_summary.to_string(index=False))

    # 4. MODELO NEURAL: BERTopic (Sentence Transformers + c-TF-IDF)
    logger.info("-" * 50)
    logger.info("ETAPA 4: MODELAGEM NEURAL COM BERTopic")
    logger.info("-" * 50)
    bertopic_pipe = BERTopicPatentPipeline()
    bertopic_pipe.fit_transform(corpus_docs, timestamps=pub_years)
    bertopic_topics = bertopic_pipe.get_topic_words_dict(top_n=10)
    df_bertopic_summary = bertopic_pipe.export_topics_table()
    bertopic_pipe.export_visualizations(corpus_docs, timestamps=pub_years)
    print("\n[TÓPICOS EXTRAÍDOS PELO BERTopic - NEURAL]:")
    print(df_bertopic_summary[["Topic", "Count", "Name"]].head(10).to_string(index=False))

    # 5. AVALIAÇÃO DE MÉTRICAS QUANTITATIVAS E RELATÓRIO
    logger.info("-" * 50)
    logger.info("ETAPA 5: AVALIAÇÃO COMPARATIVA DE MÉTRICAS (TD & UMass)")
    logger.info("-" * 50)
    df_comparison = TopicEvaluator.generate_comparison_report(
        docs=corpus_docs,
        lda_topics=lda_topics,
        bertopic_topics=bertopic_topics
    )
    print("\n" + "=" * 70)
    print("RELATÓRIO COMPARATIVO FINAL PARA O TCC (E2):")
    print("=" * 70)
    print(df_comparison.to_string(index=False))
    print("=" * 70)
    logger.info(f"Execução concluída com sucesso! Todos os artefatos foram gerados em: {OUTPUTS_DIR.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Modelagem de Tópicos em Patentes de IA (E2)")
    parser.add_argument("--input", type=str, default=None, help="Caminho para arquivo CSV/JSON de patentes")
    parser.add_argument("--sample", action="store_true", default=True, help="Usar corpus sintético calibrado")
    parser.add_argument("--size", type=int, default=500, help="Tamanho da amostra sintética (padrão: 500)")
    
    args = parser.parse_args()
    run_patent_pipeline(input_file=args.input, use_sample=args.sample, sample_size=args.size)
