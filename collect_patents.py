"""
Módulo de Coleta e Carregamento de Patentes (CPC G06N - 2019 a 2026)
Suporta dados de The Lens, PatentsView API e geração de corpus benchmark para validação.
"""

import json
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import requests

from config import CPC_SUBCLASS, START_YEAR, END_YEAR, RAW_DATA_DIR, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class PatentCollector:
    """
    Gerenciador de ingestão, filtragem e deduplicação de patentes da subclasse CPC G06N.
    """
    def __init__(self, cpc_subclass: str = CPC_SUBCLASS, start_year: int = START_YEAR, end_year: int = END_YEAR):
        self.cpc_subclass = cpc_subclass
        self.start_year = start_year
        self.end_year = end_year

    def deduplicate_by_simple_families(self, df: pd.DataFrame, family_col: str = "simple_family_id") -> pd.DataFrame:
        """
        Deduplica o corpus por Famílias Simples de Patentes (Simple Patent Families),
        evitando super-representação de depósitos redundantes em múltiplos países.
        """
        initial_count = len(df)
        if family_col in df.columns:
            df_dedup = df.drop_duplicates(subset=[family_col], keep="first").copy()
            logger.info(f"Deduplicação por família simples ({family_col}): {initial_count} -> {len(df_dedup)} documentos.")
            return df_dedup
        
        cols_to_check = [c for c in ["title", "abstract"] if c in df.columns]
        if cols_to_check:
            df_dedup = df.drop_duplicates(subset=cols_to_check, keep="first").copy()
            logger.info(f"Deduplicação por conteúdo textual (título/resumo): {initial_count} -> {len(df_dedup)} documentos.")
            return df_dedup
            
        return df

    def load_lens_export(self, file_path: Path) -> pd.DataFrame:
        """Carrega e padroniza arquivo CSV ou JSON exportado do The Lens."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
        
        if file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() in [".json", ".jsonl"]:
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Formato não suportado: {file_path.suffix}")

        logger.info(f"Carregados {len(df)} registros brutos de {file_path.name}")
        
        column_mapping = {
            "Lens ID": "patent_id",
            "Title": "title",
            "Abstract": "abstract",
            "Publication Year": "publication_year",
            "Publication Date": "publication_date",
            "Simple Family ID": "simple_family_id",
            "Applicants": "assignee",
            "CPC Classifications": "cpc_classes"
        }
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        if "publication_year" in df.columns:
            df = df[(df["publication_year"] >= self.start_year) & (df["publication_year"] <= self.end_year)]
            
        df = self.deduplicate_by_simple_families(df)
        return df

    def fetch_patentsview_sample(self, max_records: int = 500) -> pd.DataFrame:
        """
        Consulta a API PatentsView para coletar patentes concedidas ou publicadas no CPC G06N.
        """
        url = "https://api.patentsview.org/patents/query"
        query = {
            "_and": [
                {"cpc_subgroup_id": f"{self.cpc_subclass}"},
                {"_gte": {"patent_date": f"{self.start_year}-01-01"}},
                {"_lte": {"patent_date": f"{self.end_year}-12-31"}}
            ]
        }
        fields = [
            "patent_id", "patent_number", "patent_title", "patent_abstract",
            "patent_date", "patent_year", "assignee_organization"
        ]
        params = {
            "q": json.dumps(query),
            "f": json.dumps(fields),
            "o": json.dumps({"size": min(max_records, 1000)})
        }
        
        try:
            logger.info(f"Consultando PatentsView API para classe {self.cpc_subclass}...")
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            patents = data.get("patents", [])
            if not patents:
                logger.warning("Nenhum registro retornado da PatentsView API.")
                return pd.DataFrame()
            
            records = []
            for p in patents:
                assignees = p.get("assignees", [])
                assignee_org = assignees[0].get("assignee_organization") if assignees else "Unknown"
                records.append({
                    "patent_id": p.get("patent_id"),
                    "title": p.get("patent_title", ""),
                    "abstract": p.get("patent_abstract", ""),
                    "publication_date": p.get("patent_date"),
                    "publication_year": int(p.get("patent_year", 2022)),
                    "assignee": assignee_org,
                    "cpc_subclass": self.cpc_subclass,
                    "simple_family_id": p.get("patent_id")
                })
            df = pd.DataFrame(records)
            return self.deduplicate_by_simple_families(df)
        except Exception as e:
            logger.error(f"Erro ao consultar PatentsView API: {e}")
            return pd.DataFrame()

    def generate_benchmark_corpus(self, n_samples: int = 500) -> pd.DataFrame:
        """
        Gera um corpus benchmark controlado de alta fidelidade com resumos representativos
        de patentes reais da subclasse CPC G06N (2019-2026), cobrindo os clusters seminais:
        1. Transformer Models & Large Language Models (LLMs)
        2. Convolutional Neural Networks & Computer Vision
        3. Reinforcement Learning & Autonomous Decision Systems
        4. Neuromorphic Computing & Hardware Acceleration
        5. Quantum Machine Learning & Quantum Circuits
        6. Federated Learning, Privacy-Preserving AI & Differential Privacy
        7. Generative Adversarial Networks (GANs) & Diffusion Synthesis
        8. Graph Neural Networks (GNNs) & Relational Knowledge Graphs
        """
        logger.info(f"Gerando corpus benchmark sintético/calibrado com {n_samples} patentes CPC G06N...")
        
        clusters = [
            {
                "topic_label": "Transformers and Large Language Models",
                "cpc": "G06N 3/04",
                "assignees": ["Google LLC", "Microsoft Technology Licensing", "OpenAI LLC", "Meta Platforms Inc.", "Amazon Technologies"],
                "templates": [
                    "A method for multi-head self-attention mechanisms in transformer neural networks. The system comprising cross-attention encoders and auto-regressive decoders configured to process tokenized contextual embeddings for generative sequence-to-sequence language modeling.",
                    "An apparatus comprising deep transformer architectures utilizing masked language representation and positional encodings. Wherein the attention weights are sparsified using low-rank matrix decomposition to accelerate inference in massive neural language models.",
                    "A computer-implemented method for fine-tuning pre-trained generative foundation models. The system optimizes parameter-efficient adapters including low-rank adaptation LoRA and soft prompting vectors for dialogue agents and semantic text synthesis."
                ]
            },
            {
                "topic_label": "Computer Vision and Deep Convolutional Networks",
                "cpc": "G06N 3/08",
                "assignees": ["Sony Group Corp", "Samsung Electronics", "Apple Inc.", "Intel Corp", "Tencent Technology"],
                "templates": [
                    "A deep convolutional neural network system configured for multi-scale object detection and semantic image segmentation. The architecture comprises residual connection blocks, depthwise separable convolutions, and feature pyramid pooling layers.",
                    "A method for real-time video spatial-temporal feature extraction. The convolutional layers process sequential frame tensors, wherein a temporal attention gate aggregates motion vectors for automated visual surveillance and anomaly detection.",
                    "An image classification apparatus comprising convolutional layers coupled with vision transformer patches. The network computes localized gradient backpropagation to optimize receptive fields for biometric identification and medical imaging."
                ]
            },
            {
                "topic_label": "Reinforcement Learning and Autonomous Control",
                "cpc": "G06N 3/092",
                "assignees": ["Waymo LLC", "Tesla Inc.", "DeepMind Technologies", "Baidu Inc.", "Toyota Motor Corp"],
                "templates": [
                    "A deep reinforcement learning control system for autonomous vehicle trajectory planning. The agent executes actor-critic policy gradients utilizing temporal difference learning and experience replay buffers to maximize reward in continuous action spaces.",
                    "A robotic manipulator control apparatus comprising deep Q-networks DQN and model-predictive path integrals. Wherein the neural policy network maps high-dimensional LiDAR state observations into kinematic torque commands with safety constraints.",
                    "A multi-agent reinforcement learning optimization method for industrial microgrid dispatch. Decentralized actors interact through cooperative game-theoretic policies, wherein value function estimators minimize distribution loss."
                ]
            },
            {
                "topic_label": "Neuromorphic Computing and Spiking Neural Hardware",
                "cpc": "G06N 3/063",
                "assignees": ["Intel Corp", "IBM Corp", "Qualcomm Inc.", "CEA Tech", "Applied Materials Inc."],
                "templates": [
                    "A neuromorphic processor comprising an asynchronous array of spiking neuron circuits and crossbar memristor synaptic cross-points. The synaptic weights update via spike-timing-dependent plasticity STDP with low static power dissipation.",
                    "An ultra-low power event-driven spiking neural network accelerator. The integrated circuit comprises digital integrate-and-fire nodes communicating through an address-event representation AER bus for edge sensor processing.",
                    "A non-volatile memory synaptic crossbar array for hardware artificial neural networks. Memristive phase-change devices execute vector-matrix multiplication in analog current domain with in-memory computing primitives."
                ]
            },
            {
                "topic_label": "Quantum Machine Learning and Computing",
                "cpc": "G06N 10/00",
                "assignees": ["IBM Corp", "Rigetti Computing", "Google LLC", "D-Wave Systems", "Honeywell International"],
                "templates": [
                    "A hybrid quantum-classical neural network optimization system. The apparatus executes parameterized quantum circuits PQC on superconducting qubit registers, wherein classical gradient descent updates rotation angle parameters.",
                    "A quantum machine learning method utilizing quantum kernel estimation and variational quantum eigensolvers. Superconducting transmon qubits evaluate Hilbert space distance metrics for high-dimensional support vector classification.",
                    "A quantum annealer system configured for combinatorial optimization in computational graph partitioning. Superconducting flux qubits encode Ising spin Hamiltonian coupling matrices to accelerate training convergence."
                ]
            },
            {
                "topic_label": "Federated Learning and Differential Privacy",
                "cpc": "G06N 20/00",
                "assignees": ["Apple Inc.", "Google LLC", "Siemens AG", "SAP SE", "Alibaba Group"],
                "templates": [
                    "A privacy-preserving federated learning framework for distributed edge devices. The aggregation server executes federated averaging FedAvg on encrypted local model weight gradients with differential privacy Gaussian noise injection.",
                    "A secure decentralized machine learning method comprising secure multi-party computation SMPC and homomorphic encryption. Edge mobile devices train local models without transmitting raw user training datasets to a central repository.",
                    "A Byzantine-tolerant federated learning system configured to detect poisoned gradient updates from compromised edge nodes. The server calculates geometric median weight vectors across distributed participants."
                ]
            },
            {
                "topic_label": "Generative Adversarial Networks and Diffusion Synthesis",
                "cpc": "G06N 3/04",
                "assignees": ["NVIDIA Corp", "Adobe Inc.", "Runway AI", "ByteDance Ltd", "Snap Inc."],
                "templates": [
                    "A denoising score-based diffusion probabilistic model for high-resolution image synthesis. The generative neural network progressively predicts noise residuals across forward Gaussian schedules conditioned on semantic text prompts.",
                    "A conditional generative adversarial network GAN comprising generator and discriminator networks in a minimax game. The generator synthesizes photorealistic textures while a patch discriminator penalizes synthetic artifacts.",
                    "A latent diffusion architecture comprising variational autoencoders VAE and cross-attention spatial transformers. The latent space representations undergo iterative Markov chain reverse diffusion for 3D asset generation."
                ]
            },
            {
                "topic_label": "Graph Neural Networks and Knowledge Graphs",
                "cpc": "G06N 5/02",
                "assignees": ["Meta Platforms Inc.", "Amazon Technologies", "LinkedIn Corp", "NEC Corp", "Huawei Technologies"],
                "templates": [
                    "A graph convolutional neural network GCN system for relational entity prediction in heterogeneous knowledge graphs. Message passing layers aggregate topological node neighborhood embeddings across multi-hop edges.",
                    "A graph attention network GAT apparatus configured for molecular property prediction and drug discovery. The neural layers calculate dynamic attention weights between bonded atomic nodes in chemical graph structures.",
                    "A scalable distributed graph neural network training pipeline. Partitioned subgraph mini-batches undergo neighborhood sampling and tensor aggregation to infer latent connections in enterprise social networks."
                ]
            }
        ]

        random.seed(42)
        records = []
        base_date = datetime(self.start_year, 1, 15)
        total_days = (datetime(self.end_year, 10, 1) - base_date).days
        
        for i in range(n_samples):
            cluster = random.choice(clusters)
            template = random.choice(cluster["templates"])
            assignee = random.choice(cluster["assignees"])
            
            variation = random.choice([
                " In another aspect, the computing system executes matrix multipliers.",
                " Wherein the neural weights are quantized to int8 precision.",
                " The apparatus further comprises hardware accelerator engines.",
                " The model parameters are updated via stochastic gradient descent.",
                " Wherein real-time inference latency is substantially reduced."
            ])
            abstract_text = template + variation
            
            random_days = random.randint(0, total_days)
            pub_date = base_date + timedelta(days=random_days)
            pub_year = pub_date.year
            
            records.append({
                "patent_id": f"US-{pub_year}-{100000 + i}",
                "title": f"{cluster['topic_label']} for Technical Computing Arrangements",
                "abstract": abstract_text,
                "publication_date": pub_date.strftime("%Y-%m-%d"),
                "publication_year": pub_year,
                "assignee": assignee,
                "cpc_subclass": cluster["cpc"],
                "simple_family_id": f"FAM-{pub_year}-{10000 + (i // 2)}",
                "ground_truth_cluster": cluster["topic_label"]
            })

        df = pd.DataFrame(records)
        df = self.deduplicate_by_simple_families(df)
        
        output_file = RAW_DATA_DIR / "corpus_patentes_cpc_g06n_benchmark.csv"
        df.to_csv(output_file, index=False, encoding="utf-8")
        logger.info(f"Corpus benchmark salvo com sucesso em: {output_file.name} ({len(df)} registros)")
        return df
