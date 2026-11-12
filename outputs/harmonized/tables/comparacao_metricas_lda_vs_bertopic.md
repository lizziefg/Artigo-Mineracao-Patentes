# Comparação de Métricas: LDA (Baseline) vs BERTopic (Neural)

| Modelo                                              |   Número de Tópicos (K) |   Topic Diversity (TD) |   Coerência Temática (UMass) | Capacidade Contextual                | Interpretabilidade            | Tratamento de Ruído               |
|:----------------------------------------------------|------------------------:|-----------------------:|-----------------------------:|:-------------------------------------|:------------------------------|:----------------------------------|
| Baseline: Latent Dirichlet Allocation (LDA)         |                       9 |                 0.8444 |                      -2.4957 | Bag of Words (Ausência de Semântica) | Média (Sensível a Frequência) | Estático (via Stopwords)          |
| Neural: BERTopic (Sentence Transformers + c-TF-IDF) |                       9 |                 0.8111 |                      -2.3872 | Alta (Embeddings Densos)             | Alta (c-TF-IDF por Cluster)   | Dinâmico (Isolamento via HDBSCAN) |

*Nota: Valores de Topic Diversity mais próximos de 1.0 indicam menor redundância vocabular. Valores de UMass mais próximos de 0.0 indicam maior coerência contextual intramodelo.*
