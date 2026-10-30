"""
Módulo de Avaliação Quantitativa e Métricas Comparativas (Sprint E2)
Calcula Topic Diversity (TD) e Coerência Temática (UMass), gerando tabelas para o TCC.
"""

import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

from config import TABLES_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class TopicEvaluator:
    """
    Avalia e compara quantitativamente os modelos LDA e BERTopic
    em termos de dispersão lexical e coerência semântica.
    """
    @staticmethod
    def calculate_topic_diversity(topics_dict: Dict[int, List[str]], top_n: int = 10) -> float:
        """
        Calcula a Diversidade de Tópicos (Topic Diversity - TD):
        TD = len(set(todas as palavras top_n)) / (K * top_n)
        """
        all_words = []
        for words in topics_dict.values():
            all_words.extend(words[:top_n])
        if not all_words:
            return 0.0
        return float(len(set(all_words)) / len(all_words))

    @staticmethod
    def calculate_umass_coherence(docs: List[str], topics_dict: Dict[int, List[str]], top_n: int = 10) -> float:
        """
        Calcula o escore de Coerência UMass intrínseco:
        C_UMass(V) = (2 / (M * (M - 1))) * sum_{m=2}^M sum_{l=1}^{m-1} log( (D(v_m, v_l) + 1) / D(v_l) )
        """
        tokenized_docs = [set(d.lower().split()) for d in docs]
        total_docs = len(tokenized_docs)
        if total_docs == 0:
            return 0.0

        scores = []
        for tid, words in topics_dict.items():
            top_words = words[:top_n]
            m_len = len(top_words)
            if m_len < 2:
                continue

            topic_score = 0.0
            num_pairs = 0
            for m in range(1, m_len):
                w_m = top_words[m]
                d_m = sum(1 for doc in tokenized_docs if w_m in doc)
                for l in range(0, m):
                    w_l = top_words[l]
                    d_l = sum(1 for doc in tokenized_docs if w_l in doc)
                    d_ml = sum(1 for doc in tokenized_docs if w_m in doc and w_l in doc)
                    
                    if d_l > 0:
                        topic_score += np.log((d_ml + 1.0) / d_l)
                        num_pairs += 1
                        
            if num_pairs > 0:
                scores.append(topic_score / num_pairs)

        return float(np.mean(scores)) if scores else 0.0

    @classmethod
    def generate_comparison_report(
        cls,
        docs: List[str],
        lda_topics: Dict[int, List[str]],
        bertopic_topics: Dict[int, List[str]],
        output_filename: str = "comparacao_metricas_lda_vs_bertopic.csv"
    ) -> pd.DataFrame:
        """Gera a tabela comparativa oficial entre LDA e BERTopic para o TCC."""
        logger.info("Calculando métricas comparativas entre LDA e BERTopic...")
        
        lda_td = cls.calculate_topic_diversity(lda_topics, top_n=10)
        lda_umass = cls.calculate_umass_coherence(docs, lda_topics, top_n=10)
        
        bertopic_td = cls.calculate_topic_diversity(bertopic_topics, top_n=10)
        bertopic_umass = cls.calculate_umass_coherence(docs, bertopic_topics, top_n=10)
        
        comparison_data = [
            {
                "Modelo": "Baseline: Latent Dirichlet Allocation (LDA)",
                "Número de Tópicos (K)": len(lda_topics),
                "Topic Diversity (TD)": round(lda_td, 4),
                "Coerência Temática (UMass)": round(lda_umass, 4),
                "Capacidade Contextual": "Bag of Words (Ausência de Semântica)",
                "Interpretabilidade": "Média (Sensível a Frequência)",
                "Tratamento de Ruído": "Estático (via Stopwords)"
            },
            {
                "Modelo": "Neural: BERTopic (Sentence Transformers + c-TF-IDF)",
                "Número de Tópicos (K)": len(bertopic_topics),
                "Topic Diversity (TD)": round(bertopic_td, 4),
                "Coerência Temática (UMass)": round(bertopic_umass, 4),
                "Capacidade Contextual": "Alta (Embeddings Densos)",
                "Interpretabilidade": "Alta (c-TF-IDF por Cluster)",
                "Tratamento de Ruído": "Dinâmico (Isolamento via HDBSCAN)"
            }
        ]
        
        df_comp = pd.DataFrame(comparison_data)
        csv_path = TABLES_DIR / output_filename
        df_comp.to_csv(csv_path, index=False, encoding="utf-8-sig")
        
        md_path = TABLES_DIR / output_filename.replace(".csv", ".md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Comparação de Métricas: LDA (Baseline) vs BERTopic (Neural)\n\n")
            f.write(df_comp.to_markdown(index=False))
            f.write("\n\n*Nota: Valores de Topic Diversity mais próximos de 1.0 indicam menor redundância vocabular. Valores de UMass mais próximos de 0.0 indicam maior coerência contextual intramodelo.*\n")
            
        logger.info(f"Relatório comparativo salvo em:\n  - CSV: {csv_path.name}\n  - Markdown: {md_path.name}")
        return df_comp
