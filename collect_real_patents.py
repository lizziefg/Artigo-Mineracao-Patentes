"""
Módulo de Coleta e Padronização de Patentes Reais (CPC G06N, 2019–2026).
Sprint E3 - TCC CST em Inteligência Artificial - FATEC Rio Claro.

Suporta exclusivamente dados reais provenientes de duas fontes oficiais:
1. USPTO Open Data Portal (ODP / legado PatentsView): API v1 ou exportação local CSV/JSON.
2. The Lens (Cambia): Patent API v1 ou exportação direta CSV/JSONL/JSON do portal web.
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import requests

from config import (
    API_CONFIG,
    CPC_SUBCLASS,
    END_YEAR,
    PROCESSED_DATA_DIR,
    RAW_LENS_DIR,
    RAW_USPTO_DIR,
    START_YEAR,
    get_api_credential,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STANDARD_COLUMNS = [
    "patent_id",
    "title",
    "abstract",
    "publication_date",
    "publication_year",
    "assignee",
    "assignee_country",
    "jurisdiction",
    "cpc_subclass",
    "simple_family_id",
    "source",
]


def _extract_year(date_str: Any) -> Optional[int]:
    """Extrai o ano (YYYY) de uma data ou string numérica de publicação."""
    if pd.isna(date_str):
        return None
    s = str(date_str).strip()
    if len(s) >= 4 and s[:4].isdigit():
        return int(s[:4])
    return None


def _compute_text_family_fallback(title: str, abstract: str) -> str:
    """Gera chave determinística de deduplicação quando Simple Family ID estiver ausente."""
    norm = f"{str(title).strip().lower()}||{str(abstract).strip().lower()}"
    return "HASH-" + hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def finalize_and_deduplicate(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """
    Aplica filtro temporal (2019–2026), exige resumo válido e deduplica por Simple Patent Family.
    """
    if df.empty:
        logger.warning(f"[{source_label}] DataFrame vazio antes da higienização.")
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    df = df.copy()
    for col in STANDARD_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df["source"] = source_label
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["abstract"] = df["abstract"].fillna("").astype(str).str.strip()
    df["assignee"] = df["assignee"].fillna("Não Informado").replace("", "Não Informado").astype(str).str.strip()
    df["assignee_country"] = (
        df["assignee_country"].fillna("XX").replace("", "XX").astype(str).str.strip().str.upper()
    )
    df["jurisdiction"] = df["jurisdiction"].fillna("XX").replace("", "XX").astype(str).str.strip().str.upper()

    # Extração e filtro temporal
    if df["publication_year"].isna().any() or (df["publication_year"] == "").any():
        df["publication_year"] = df["publication_year"].fillna(df["publication_date"].apply(_extract_year))
    df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce")
    df = df.dropna(subset=["publication_year"]).copy()
    df["publication_year"] = df["publication_year"].astype(int)
    df = df[(df["publication_year"] >= START_YEAR) & (df["publication_year"] <= END_YEAR)].copy()

    # Exigência estrita de resumo não vazio para mineração de tópicos
    df = df[df["abstract"].str.len() >= 30].copy()

    # Preencher simple_family_id com fallback determinístico apenas quando ausente
    missing_fam = df["simple_family_id"].isna() | (df["simple_family_id"].astype(str).str.strip() == "")
    if missing_fam.any():
        df.loc[missing_fam, "simple_family_id"] = df.loc[missing_fam].apply(
            lambda row: _compute_text_family_fallback(row["title"], row["abstract"]), axis=1
        )

    before = len(df)
    df = df.sort_values("publication_date", ascending=True).drop_duplicates(
        subset=["simple_family_id"], keep="first"
    )
    # Segunda camada de deduplicação exata por título+resumo normalizados
    df["_text_sig"] = df.apply(lambda r: _compute_text_family_fallback(r["title"], r["abstract"]), axis=1)
    df = df.drop_duplicates(subset=["_text_sig"], keep="first").drop(columns=["_text_sig"])

    logger.info(
        f"[{source_label}] Deduplicação por família simples + assinatura textual: {before} -> {len(df)} famílias."
    )
    return df[STANDARD_COLUMNS].reset_index(drop=True)


class USPTOOpenDataCollector:
    """
    Coletor de dados reais de patentes norte-americanas via USPTO Open Data Portal (ODP)
    ou arquivos locais exportados do portal data.uspto.gov / legado PatentsView.
    """

    def __init__(self, api_key: Optional[str] = None, cpc_subclass: str = CPC_SUBCLASS):
        self.api_key = get_api_credential(API_CONFIG["uspto_api_key_env"], api_key)
        self.cpc_subclass = cpc_subclass

    def fetch_from_api(self, max_records: int = 2000, page_size: int = 100) -> pd.DataFrame:
        """
        Consulta a API oficial do USPTO Open Data Portal (/api/v1/patent/applications/search)
        filtrando pedidos com classificação CPC iniciando em G06N no período 2019–2026.
        """
        if not self.api_key:
            raise PermissionError(
                "USPTO_API_KEY não informada. Configure a variável de ambiente USPTO_API_KEY, "
                "passe --uspto-key ou forneça uma exportação local CSV/JSON via --uspto-file."
            )

        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        records: List[Dict[str, Any]] = []
        offset = 0

        logger.info(f"[USPTO_ODP] Iniciando coleta via API (alvo: até {max_records} registros em {self.cpc_subclass})...")
        while offset < max_records:
            batch_limit = min(page_size, max_records - offset)
            payload = {
                "q": f'applicationMetaData.cpcClassificationBag:{self.cpc_subclass}* AND applicationMetaData.earliestPublicationDate:[{START_YEAR}-01-01 TO {END_YEAR}-12-31]',
                "pagination": {"offset": offset, "limit": batch_limit},
            }
            resp = requests.post(
                API_CONFIG["uspto_odp_url"],
                headers=headers,
                json=payload,
                timeout=API_CONFIG["request_timeout"],
            )
            resp.raise_for_status()
            data = resp.json()
            bag = data.get("patentFileWrapperDataBag", [])
            if not bag:
                break

            for item in bag:
                meta = item.get("applicationMetaData", {})
                pub_date = meta.get("earliestPublicationDate") or meta.get("filingDate") or ""
                pub_num = meta.get("earliestPublicationNumber") or item.get("applicationNumberText", "")

                applicants = meta.get("applicantBag", []) or []
                first_app = applicants[0] if applicants else {}
                assignee = first_app.get("applicantNameText") or meta.get("firstApplicantName") or "Não Informado"
                country = (
                    first_app.get("correspondenceAddressBag", [{}])[0].get("countryCode", "US")
                    if first_app.get("correspondenceAddressBag")
                    else first_app.get("countryCode", "US")
                )

                cpc_list = meta.get("cpcClassificationBag", []) or []
                cpc_str = ";".join([str(c) for c in cpc_list if str(c).startswith(self.cpc_subclass)]) or self.cpc_subclass

                abstract_text = item.get("abstractText") or meta.get("abstractText") or ""
                records.append(
                    {
                        "patent_id": pub_num,
                        "title": meta.get("inventionTitle", ""),
                        "abstract": abstract_text,
                        "publication_date": pub_date,
                        "publication_year": _extract_year(pub_date),
                        "assignee": assignee,
                        "assignee_country": country or "US",
                        "jurisdiction": "US",
                        "cpc_subclass": cpc_str,
                        "simple_family_id": item.get("applicationNumberText") or pub_num,
                    }
                )

            offset += len(bag)
            logger.info(f"[USPTO_ODP] Coletados {len(records)} registros até o momento...")
            if len(bag) < batch_limit:
                break
            time.sleep(0.3)

        raw_out = RAW_USPTO_DIR / "uspto_odp_raw.json"
        with open(raw_out, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        logger.info(f"[USPTO_ODP] Snapshot bruto salvo em: {raw_out}")

        return finalize_and_deduplicate(pd.DataFrame(records), "USPTO_ODP")

    def load_local_export(self, file_path: Path) -> pd.DataFrame:
        """Carrega exportação real local (CSV, TSV ou JSON) do USPTO Open Data Portal / PatentsView."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Arquivo USPTO não encontrado: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(file_path, low_memory=False)
        elif suffix == ".tsv":
            df = pd.read_csv(file_path, sep="\t", low_memory=False)
        elif suffix in {".json", ".jsonl"}:
            df = pd.read_json(file_path, lines=(suffix == ".jsonl"))
        else:
            raise ValueError(f"Formato não suportado para arquivo USPTO: {suffix}")

        col_map = {
            "patent_number": "patent_id",
            "publicationNumber": "patent_id",
            "earliestPublicationNumber": "patent_id",
            "publication_number": "patent_id",
            "Document ID": "patent_id",
            "Publication/Patent Number": "patent_id",
            "patent_title": "title",
            "inventionTitle": "title",
            "Title": "title",
            "patent_abstract": "abstract",
            "abstractText": "abstract",
            "Abstract": "abstract",
            "patent_date": "publication_date",
            "earliestPublicationDate": "publication_date",
            "Date Published": "publication_date",
            "Publication Date": "publication_date",
            "patent_year": "publication_year",
            "assignee_organization": "assignee",
            "applicantName": "assignee",
            "Assignee": "assignee",
            "Applicant": "assignee",
            "assignee_name": "assignee",
            "assignee_country": "assignee_country",
            "applicantCountry": "assignee_country",
            "country_code": "jurisdiction",
            "cpc_subgroup_id": "cpc_subclass",
            "cpcClassification": "cpc_subclass",
            "cpc_codes": "cpc_subclass",
            "CPC": "cpc_subclass",
            "application_number": "simple_family_id",
            "Application Number": "simple_family_id",
            "family_id": "simple_family_id",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        if "jurisdiction" not in df.columns:
            df["jurisdiction"] = "US"
        if "assignee_country" not in df.columns:
            df["assignee_country"] = "US"
        if "cpc_subclass" not in df.columns:
            df["cpc_subclass"] = self.cpc_subclass

        return finalize_and_deduplicate(df, "USPTO_ODP")


class LensPatentCollector:
    """
    Coletor de dados reais de patentes internacionais via The Lens (Cambia),
    suportando a Patent API oficial ou arquivos CSV/JSON exportados do portal web.
    """

    def __init__(self, api_token: Optional[str] = None, cpc_subclass: str = CPC_SUBCLASS):
        self.api_token = get_api_credential(API_CONFIG["lens_api_token_env"], api_token)
        self.cpc_subclass = cpc_subclass

    def fetch_from_api(self, max_records: int = 2000, page_size: int = 100) -> pd.DataFrame:
        """
        Consulta a API oficial do The Lens (/patent/search) usando paginação por scroll/offset
        para patentes classificadas em CPC G06N* publicadas entre 2019 e 2026.
        """
        if not self.api_token:
            raise PermissionError(
                "LENS_API_TOKEN não informado. Configure a variável de ambiente LENS_API_TOKEN, "
                "passe --lens-token ou exporte um arquivo CSV/JSON pelo site lens.org e use --lens-file."
            )

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }
        records: List[Dict[str, Any]] = []
        scroll_id: Optional[str] = None

        logger.info(f"[THE_LENS] Iniciando coleta via API (alvo: até {max_records} registros em {self.cpc_subclass})...")
        while len(records) < max_records:
            body: Dict[str, Any] = {
                "query": {
                    "bool": {
                        "must": [
                            {"prefix": {"class_cpc.symbol": self.cpc_subclass}},
                            {"range": {"date_published": {"gte": f"{START_YEAR}-01-01", "lte": f"{END_YEAR}-12-31"}}},
                            {"term": {"has_abstract": True}},
                        ]
                    }
                },
                "size": min(page_size, max_records - len(records)),
                "include": [
                    "lens_id",
                    "jurisdiction",
                    "doc_number",
                    "kind",
                    "date_published",
                    "biblio.invention_title",
                    "abstract",
                    "biblio.parties.applicants",
                    "biblio.parties.owners_all",
                    "biblio.classifications_cpc",
                    "families.simple_family",
                ],
                "scroll": "2m",
            }
            if scroll_id:
                body = {"scroll_id": scroll_id, "scroll": "2m"}

            resp = requests.post(
                API_CONFIG["lens_api_url"],
                headers=headers,
                json=body,
                timeout=API_CONFIG["request_timeout"],
            )
            resp.raise_for_status()
            payload = resp.json()
            scroll_id = payload.get("scroll_id")
            hits = payload.get("data", [])
            if not hits:
                break

            for doc in hits:
                biblio = doc.get("biblio", {})
                titles = biblio.get("invention_title", []) or []
                en_title = next((t.get("text", "") for t in titles if t.get("lang") == "en"), "")
                title_text = en_title or (titles[0].get("text", "") if titles else "")

                abstracts = doc.get("abstract", []) or []
                if isinstance(abstracts, list):
                    en_abs = next((a.get("text", "") for a in abstracts if a.get("lang") == "en"), "")
                    abs_text = en_abs or (abstracts[0].get("text", "") if abstracts else "")
                else:
                    abs_text = str(abstracts)

                parties = biblio.get("parties", {})
                applicants = parties.get("applicants", []) or parties.get("owners_all", []) or []
                first_app = applicants[0] if applicants else {}
                assignee = first_app.get("extracted_name", {}).get("value") or "Não Informado"
                app_country = first_app.get("residence") or doc.get("jurisdiction", "XX")

                cpc_items = biblio.get("classifications_cpc", {}).get("classifications", []) or []
                cpc_str = ";".join(
                    [c.get("symbol", "") for c in cpc_items if str(c.get("symbol", "")).startswith(self.cpc_subclass)]
                ) or self.cpc_subclass

                fam_members = doc.get("families", {}).get("simple_family", {}).get("members", []) or []
                fam_id = fam_members[0].get("document_id", {}).get("doc_number") if fam_members else doc.get("lens_id")

                pub_date = doc.get("date_published", "")
                pub_num = f"{doc.get('jurisdiction', '')}{doc.get('doc_number', '')}{doc.get('kind', '')}"

                records.append(
                    {
                        "patent_id": pub_num or doc.get("lens_id"),
                        "title": title_text,
                        "abstract": abs_text,
                        "publication_date": pub_date,
                        "publication_year": _extract_year(pub_date),
                        "assignee": assignee,
                        "assignee_country": app_country,
                        "jurisdiction": doc.get("jurisdiction", "XX"),
                        "cpc_subclass": cpc_str,
                        "simple_family_id": fam_id or doc.get("lens_id"),
                    }
                )

            logger.info(f"[THE_LENS] Coletados {len(records)} registros até o momento...")
            if not scroll_id or len(hits) == 0:
                break
            time.sleep(0.3)

        raw_out = RAW_LENS_DIR / "the_lens_raw.json"
        with open(raw_out, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        logger.info(f"[THE_LENS] Snapshot bruto salvo em: {raw_out}")

        return finalize_and_deduplicate(pd.DataFrame(records), "THE_LENS")

    def load_local_export(self, file_path: Path) -> pd.DataFrame:
        """
        Carrega exportação oficial em CSV, JSON ou JSONL baixada diretamente do portal The Lens.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Arquivo The Lens não encontrado: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(file_path, low_memory=False)
        elif suffix in {".json", ".jsonl"}:
            df = pd.read_json(file_path, lines=(suffix == ".jsonl"))
        else:
            raise ValueError(f"Formato não suportado para arquivo The Lens: {suffix}")

        col_map = {
            "Lens ID": "lens_id",
            "Display Key": "patent_id",
            "Publication Number": "patent_id",
            "Title": "title",
            "Abstract": "abstract",
            "Publication Date": "publication_date",
            "Publication Year": "publication_year",
            "Applicants": "assignee",
            "Owners": "assignee_owner",
            "Jurisdiction": "jurisdiction",
            "CPC Classifications": "cpc_subclass",
            "Simple Family ID": "simple_family_id",
            "Simple Family Members": "simple_family_members",
            "Simple Family Size": "simple_family_size",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        if "patent_id" not in df.columns and "lens_id" in df.columns:
            df["patent_id"] = df["lens_id"]
        if "simple_family_id" not in df.columns and "simple_family_members" in df.columns:
            df["simple_family_id"] = (
                df["simple_family_members"]
                .fillna("")
                .astype(str)
                .apply(lambda s: "|".join(sorted(m.strip() for m in s.split(";;") if m.strip())))
            )
        if "assignee" not in df.columns and "assignee_owner" in df.columns:
            df["assignee"] = df["assignee_owner"]
        # Se houver múltiplos depositantes separados por ";;", retém o titular principal para agregação
        if "assignee" in df.columns:
            df["assignee"] = df["assignee"].fillna("").astype(str).apply(lambda s: s.split(";;")[0].strip())

        # Extrair país do depositante quando presente ou herdar jurisdição de publicação
        if "Applicant Countries" in df.columns:
            df["assignee_country"] = (
                df["Applicant Countries"].fillna("").astype(str).apply(lambda s: s.split(";;")[0].strip())
            )
        elif "assignee_country" not in df.columns:
            df["assignee_country"] = df.get("jurisdiction", "XX")

        return finalize_and_deduplicate(df, "THE_LENS")


def save_source_dataset(df: pd.DataFrame, filename: str) -> Path:
    """Persiste dataset padronizado na pasta data/processed."""
    out_path = PROCESSED_DATA_DIR / filename
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"Dataset salvo em: {out_path} ({len(df)} registros)")
    return out_path
