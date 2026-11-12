# Pipeline de Mineração Patentária e Modelagem de Tópicos em IA (CPC G06N)

> **Trabalho de Conclusão de Curso (TCC) — Modalidade Artigo Científico**  
> **Curso:** Curso Superior de Tecnologia em Inteligência Artificial — FATEC Rio Claro  
> **Tema:** Panorama da Propriedade Intelectual em Inteligência Artificial: Uma Abordagem Patentométrica via Mineração de Dados Textuais  
> **Autora:** Elisa Alcântara  
> **Orientador:** Prof. Dr. Julio Cesar de Lemos  

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset](https://img.shields.io/badge/Corpus-162.166%20fam%C3%ADlias-green.svg)](#-dados-e-corpus-patentário)

---

## 📖 Visão Geral

Este repositório contém o pipeline modular, determinístico e auditável de **Descoberta de Conhecimento em Textos de Patentes (KDD)** desenvolvido para o Trabalho de Conclusão de Curso. A pesquisa investiga o panorama global de apropriação tecnológica em Inteligência Artificial por meio da mineração de documentos de patentes da subclasse **CPC G06N** (*Computing arrangements based on specific computational models*) publicados entre **janeiro de 2019 e novembro de 2026**.

O estudo implementa uma avaliação comparativa entre dois paradigmas de modelagem não supervisionada de tópicos:
1. **Baseline Probabilístico:** *Latent Dirichlet Allocation* (LDA) operando sobre representações *Bag-of-Words* (BoW);
2. **Modelo Neural Modular:** *BERTopic* combinando *sentence embeddings* (`all-MiniLM-L6-v2`), redução dimensional (*UMAP*), agrupamento espacial por densidade (*HDBSCAN* no método `leaf`) e representação temática ponderada (*c-TF-IDF*).

Ambos os modelos são avaliados por métricas quantitativas intrínsecas: **Diversidade Temática (*Topic Diversity* — TD)** e **Coerência Temática Intrínseca ($C_{\text{UMass}}$)** (MIMNO et al., 2011).

---

## 🗂️ Dados e Corpus Patentário

A pesquisa opera sobre um **Corpus Harmonizado de 162.166 famílias simples únicas de patentes**, consolidado a partir da integração de dois grandes repositórios públicos:

| Fonte / Acervo | Registros Extraídos | Famílias Simples Únicas | Elegíveis em Inglês (PLN) | Jurisdições | Países Titulares | Depositantes Únicos |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Google Patents / USPTO (BigQuery)** | 128.137 | 127.274 | 127.219 (100,0%) | 1 (`US`) | 100 | 19.674 |
| **The Lens (Internacional — Top 50k)** | 50.000 | 48.370 | 34.180 (70,7%) | 41 | 41* | 18.595 |
| **Corpus Harmonizado (Definitivo)** | **175.644** | **162.166** | **147.983 (91,3%)** | **41** | **103** | **34.472** |

* O arquivo consolidado completo está disponibilizado compactado em [`data/processed/patentes_g06n_harmonizado.zip`](data/processed/patentes_g06n_harmonizado.zip) (46,6 MB, descompacta em 166,8 MB com as 162.166 famílias).
* As amostras estratificadas de $n = 20.000$ famílias utilizadas no treinamento dos modelos estão salvas em `data/processed/processado_harmonized.csv` e `data/processed/processado_uspto_odp.csv`.

---

## 🛠️ Estrutura do Repositório

```
Artigo-Mineracao-Patentes/
├── data/
│   ├── ne_110m_countries.geojson     # Fronteiras globais para o mapa coroplético
│   └── processed/
│       ├── patentes_g06n_harmonizado.zip  # Corpus unificado completo (162.166 famílias)
│       ├── processado_harmonized.csv      # Amostra estratificada PLN (n = 20.000)
│       └── processado_uspto_odp.csv       # Amostra estratificada USPTO (n = 20.000)
├── outputs/
│   ├── harmonized/                   # Resultados do Corpus Harmonizado
│   │   ├── figures/                  # Figuras PNG (300 DPI) e interativas (HTML Plotly)
│   │   └── tables/                   # Tabelas CSV e relatórios Markdown
│   ├── uspto_odp/                    # Resultados isolados do acervo USPTO
│   └── the_lens/                     # Resultados isolados da base The Lens
├── legacy_e2/                        # Protótipo preliminar da Sprint E2 (benchmark sintético)
├── config.py                         # Parâmetros globais, caminhos e stopwords de jargão
├── collect_real_patents.py           # Coleta automatizada nas APIs / BigQuery / The Lens
├── compare_sources.py                # Harmonização e deduplicação inter-bases por famílias
├── preprocess.py                     # Filtro de idioma, limpeza e remoção de jargão jurídico
├── lda_baseline.py                   # Implementação do modelo baseline LDA
├── bertopic_pipeline.py              # Pipeline modular BERTopic (Sentence-BERT + UMAP + HDBSCAN + c-TF-IDF)
├── evaluate_metrics.py               # Cálculo de Topic Diversity (TD) e Coerência UMass
├── visualize_e3.py                   # Geração de gráficos estáticos de alta resolução (300 DPI)
├── run_pipeline_e3.py                # Orquestrador da Sprint E3 (execução determinística)
├── requirements.txt                  # Dependências do projeto
├── LICENSE                           # Licença MIT
└── README.md                         # Documentação oficial
```

---

## 🚀 Como Reproduzir os Experimentos

### 1. Clonar o Repositório e Criar Ambiente Virtual

```bash
git clone https://github.com/lizziefg/Artigo-Mineracao-Patentes.git
cd Artigo-Mineracao-Patentes
python -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Executar o Pipeline da Sprint E3 (Dados Reais)

Para executar o pipeline completo sobre os dados processados determinísticos (`random_state = 42`):

```bash
# Execução completa para todas as fontes (USPTO, The Lens e Harmonizado):
python run_pipeline_e3.py --source all --use-processed

# Ou executar individualmente para o Corpus Harmonizado:
python run_pipeline_e3.py --source harmonized --use-processed
```

---

## 📊 Principais Resultados e Artefatos Gerados

### Tabelas (`outputs/harmonized/tables/`)
* `comparacao_metricas_lda_vs_bertopic.csv`: Avaliação intrínseca ($C_{\text{UMass}} = -2,3872$ no BERTopic vs. $-2,4957$ no LDA; $TD = 0,8111$ vs. $0,8444$).
* `bertopic_topics_summary.csv`: Resumo dos 9 clusters temáticos válidos ($K = 9$), percentuais e termos discriminantes *c-TF-IDF*.
* `evolucao_temporal_macro.csv`: Série histórica anual (2019–2026) das 162.166 famílias.
* `ranking_depositantes_macro.csv`: Maiores titulares globais (IBM, Samsung, Microsoft, Google, etc.).
* `distribuicao_paises_macro.csv`: Distribuição geográfica global de titulares (EUA 42,8%, China 28,0%, Coreia 7,3%, Japão 5,3%).

### Figuras (`outputs/harmonized/figures/`)
* `fig_clusters_barras.png`: Distribuição percentual e rótulos dos 9 clusters temáticos de IA.
* `fig_evolucao_temporal_linhas.png`: Trajetórias dinâmicas das frentes tecnológicas entre 2019 e 2026.
* `fig_mapa_paises.png`: Mapa coroplético global de densidade patentária com escala exponencial.

---

## 📄 Licença

Este projeto é disponibilizado sob a licença **MIT**. Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.
