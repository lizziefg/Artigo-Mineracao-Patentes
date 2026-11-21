"""
Módulo de Avaliação Quantitativa e Métricas Comparativas (Sprint E2)
Calcula Topic Diversity (TD) e Coerência Temática (UMass), gerando tabelas para o TCC.
"""

import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

from config import HARMONIZED_OUTPUTS_DIR

TABLES_DIR = HARMONIZED_OUTPUTS_DIR / "tables"

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

    @staticmethod
    def calculate_cv_coherence(docs: List[str], topics_dict: Dict[int, List[str]], top_n: int = 10) -> float:
        """
        Calcula a Coerência Semântica C_v (Röder, Both e Hinneburg, 2015) utilizando o Gensim.
        Combina NPMI sobre janelas deslizantes de contexto com similaridade de cosseno.
        """
        try:
            from gensim.corpora.dictionary import Dictionary
            from gensim.models.coherencemodel import CoherenceModel

            tokenized_docs = [d.lower().split() for d in docs]
            dictionary = Dictionary(tokenized_docs)
            topics = [words[:top_n] for words in topics_dict.values() if len(words) >= 2]
            if not topics:
                return 0.0
            cm = CoherenceModel(
                topics=topics,
                texts=tokenized_docs,
                dictionary=dictionary,
                coherence="c_v",
            )
            return float(cm.get_coherence())
        except Exception as e:
            logger.warning(f"Gensim indisponível ou falha no cálculo de C_v ({e}).")
            return 0.0

    @classmethod
    def generate_comparison_report(
        cls,
        docs: List[str],
        lda_topics: Dict[int, List[str]],
        bertopic_topics: Dict[int, List[str]],
        output_filename: str = "comparacao_metricas_lda_vs_bertopic.csv",
        tables_dir: Path | None = None,
    ) -> pd.DataFrame:
        """Gera a tabela comparativa oficial entre LDA e BERTopic para o TCC."""
        logger.info("Calculando métricas comparativas entre LDA e BERTopic...")
        
        lda_td = cls.calculate_topic_diversity(lda_topics, top_n=10)
        lda_umass = cls.calculate_umass_coherence(docs, lda_topics, top_n=10)
        lda_cv = cls.calculate_cv_coherence(docs, lda_topics, top_n=10)
        
        bertopic_td = cls.calculate_topic_diversity(bertopic_topics, top_n=10)
        bertopic_umass = cls.calculate_umass_coherence(docs, bertopic_topics, top_n=10)
        bertopic_cv = cls.calculate_cv_coherence(docs, bertopic_topics, top_n=10)
        
        comparison_data = [
            {
                "Modelo": "Baseline: Latent Dirichlet Allocation (LDA)",
                "Número de Tópicos (K)": len(lda_topics),
                "Coerência Semântica (C_v)": round(lda_cv, 4),
                "Coerência Intrínseca (C_UMass)": round(lda_umass, 4),
                "Topic Diversity (TD)": round(lda_td, 4),
                "Representação Espacial": "Bag of Words (Ausência de Semântica)",
                "Interpretabilidade": "Média (Sensível a Frequência)",
                "Tratamento de Ruído": "Estático (via Stopwords)"
            },
            {
                "Modelo": "Neural: BERTopic (Sentence Transformers + c-TF-IDF)",
                "Número de Tópicos (K)": len(bertopic_topics),
                "Coerência Semântica (C_v)": round(bertopic_cv, 4),
                "Coerência Intrínseca (C_UMass)": round(bertopic_umass, 4),
                "Topic Diversity (TD)": round(bertopic_td, 4),
                "Representação Espacial": "Embeddings Densos (all-MiniLM-L6-v2)",
                "Interpretabilidade": "Alta (c-TF-IDF por Cluster)",
                "Tratamento de Ruído": "Dinâmico (HDBSCAN leaf + c-TF-IDF)"
            }
        ]
        
        df_comp = pd.DataFrame(comparison_data)
        out_dir = Path(tables_dir) if tables_dir else TABLES_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / output_filename
        df_comp.to_csv(csv_path, index=False, encoding="utf-8-sig")
        
        md_path = out_dir / output_filename.replace(".csv", ".md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Comparação de Métricas: LDA (Baseline) vs BERTopic (Neural)\n\n")
            f.write(df_comp.to_markdown(index=False))
            f.write("\n\n*Nota: Valores de Topic Diversity mais próximos de 1.0 indicam menor redundância vocabular. Valores de UMass mais próximos de 0.0 indicam maior coerência contextual intramodelo.*\n")
            
        logger.info(f"Relatório comparativo salvo em:\n  - CSV: {csv_path}\n  - Markdown: {md_path}")
        return df_comp
