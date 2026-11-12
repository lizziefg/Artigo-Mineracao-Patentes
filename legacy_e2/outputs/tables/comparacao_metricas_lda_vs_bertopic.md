# Comparação de Métricas: LDA (Baseline) vs BERTopic (Neural)

| Modelo                                              |   Número de Tópicos (K) |   Topic Diversity (TD) |   Coerência Temática (UMass) | Capacidade Contextual                | Interpretabilidade            | Tratamento de Ruído               |
|:----------------------------------------------------|------------------------:|-----------------------:|-----------------------------:|:-------------------------------------|:------------------------------|:----------------------------------|
| Baseline: Latent Dirichlet Allocation (LDA)         |                       8 |                    0.7 |                      -1.2873 | Bag of Words (Ausência de Semântica) | Média (Sensível a Frequência) | Estático (via Stopwords)          |
| Neural: BERTopic (Sentence Transformers + c-TF-IDF) |                       8 |                    0.9 |                      -0.9742 | Alta (Embeddings Densos)             | Alta (c-TF-IDF por Cluster)   | Dinâmico (Isolamento via HDBSCAN) |

*Nota: Valores de Topic Diversity mais próximos de 1.0 indicam menor redundância vocabular. Valores de UMass mais próximos de 0.0 indicam maior coerência contextual intramodelo.*
