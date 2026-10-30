"""
Modelo Baseline de Modelagem de Tópicos: Latent Dirichlet Allocation (LDA)
Implementado via Scikit-Learn com cálculo de coerência intrínseca e Topic Diversity.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from config import LDA_CONFIG, TABLES_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class LDABaselineModel:
    """
    Pipeline de Latent Dirichlet Allocation (Blei et al., 2003) para extração de tópicos
    e benchmarking comparativo com o modelo neural BERTopic.
    """
    def __init__(self, n_topics: int = LDA_CONFIG["default_num_topics"], random_state: int = LDA_CONFIG["random_state"]):
        self.n_topics = n_topics
        self.random_state = random_state
        self.vectorizer = CountVectorizer(
            max_df=LDA_CONFIG["max_df"],
            min_df=LDA_CONFIG["min_df"],
            max_features=LDA_CONFIG["max_features"]
        )
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            max_iter=LDA_CONFIG["max_iter"],
            learning_method="batch",
            random_state=self.random_state,
            n_jobs=-1
        )
        self.feature_names = []
        self.fitted = False

    def fit_transform(self, texts: List[str]) -> Tuple[np.ndarray, Dict[int, List[str]]]:
        """Ajusta o vetorizador CountVectorizer e o modelo LDA ao corpus textual."""
        logger.info(f"Treinando modelo Baseline LDA com k={self.n_topics} tópicos em {len(texts)} documentos...")
        doc_term_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        doc_topic_dist = self.lda_model.fit_transform(doc_term_matrix)
        self.fitted = True
        
        topics_dict = self.get_top_words(n_words=10)
        logger.info("Treinamento do LDA concluído.")
        return doc_topic_dist, topics_dict

    def get_top_words(self, n_words: int = 10) -> Dict[int, List[str]]:
        """Retorna os top termos de cada tópico gerado pelo LDA."""
        if not self.fitted:
            raise ValueError("O modelo precisa ser treinado antes de extrair as palavras-chave.")
            
        topics = {}
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_features_ind = topic.argsort()[: -n_words - 1 : -1]
            top_features = [self.feature_names[i] for i in top_features_ind]
            topics[topic_idx] = top_features
        return topics

    def calculate_topic_diversity(self, top_words_dict: Dict[int, List[str]], top_n: int = 10) -> float:
        """
        Calcula a métrica de Diversidade de Tópicos (Topic Diversity):
        TD = |palavras únicas entre os top termos| / (K * top_n)
        Varia de 0 (tópicos redundantes/idênticos) a 1 (tópicos completamente distintos).
        """
        all_words = []
        for words in top_words_dict.values():
            all_words.extend(words[:top_n])
        if not all_words:
            return 0.0
        unique_words = len(set(all_words))
        td = unique_words / len(all_words)
        return float(td)

    def calculate_umass_coherence(self, texts: List[str], top_words_dict: Dict[int, List[str]], top_n: int = 10) -> float:
        """
        Calcula a Coerência UMass intrínseca baseada na coocorrência de termos no corpus:
        C_UMass(V) = sum_{m=2}^M sum_{l=1}^{m-1} log( (D(v_m, v_l) + 1) / D(v_l) )
        Valores mais próximos de zero indicam maior coerência temática.
        """
        tokenized_docs = [set(t.split()) for t in texts]
        total_docs = len(tokenized_docs)
        if total_docs == 0:
            return 0.0

        coherence_scores = []
        for topic_idx, words in top_words_dict.items():
            top_words = words[:top_n]
            topic_score = 0.0
            pairs_count = 0
            
            for m in range(1, len(top_words)):
                word_m = top_words[m]
                d_vm = sum(1 for doc in tokenized_docs if word_m in doc)
                for l in range(0, m):
                    word_l = top_words[l]
                    d_vl = sum(1 for doc in tokenized_docs if word_l in doc)
                    d_vm_vl = sum(1 for doc in tokenized_docs if word_m in doc and word_l in doc)
                    
                    if d_vl > 0:
                        topic_score += np.log((d_vm_vl + 1.0) / d_vl)
                        pairs_count += 1
                        
            if pairs_count > 0:
                coherence_scores.append(topic_score / pairs_count)

        return float(np.mean(coherence_scores)) if coherence_scores else 0.0

    def export_topics_summary(self, top_words_dict: Dict[int, List[str]], filename: str = "lda_topics_summary.csv") -> pd.DataFrame:
        """Salva resumo estruturado dos tópicos em CSV para tabelas do TCC."""
        records = []
        for topic_idx, words in top_words_dict.items():
            records.append({
                "Tópico ID": f"LDA_{topic_idx:02d}",
                "Top 10 Termos": ", ".join(words),
                "Termo Principal": words[0] if words else ""
            })
        df = pd.DataFrame(records)
        output_path = TABLES_DIR / filename
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info(f"Tabela de tópicos LDA salva em: {output_path.name}")
        return df
