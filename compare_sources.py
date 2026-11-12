"""
Módulo de Comparação e Compatibilização entre Fontes Reais (USPTO ODP vs. The Lens).
Sprint E3 - TCC CST em Inteligência Artificial - FATEC Rio Claro.
"""

import hashlib
import logging
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd

from config import HARMONIZED_OUTPUTS_DIR, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _normalize_id(val: str) -> str:
    """Normaliza número de publicação removendo hífens e espaços para pareamento interfontes."""
    return "".join(ch for ch in str(val).upper() if ch.isalnum())


def _normalize_title_key(title: str) -> str:
    """Gera hash de título normalizado para identificar sobreposição semântica entre bases."""
    clean = " ".join(str(title).lower().split())
    return hashlib.sha256(clean.encode("utf-8")).hexdigest()[:16]


def compare_and_harmonize(
    df_uspto: pd.DataFrame,
    df_lens: pd.DataFrame,
    output_tables_dir: Path | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compara métricas de cobertura entre USPTO ODP e The Lens e gera um corpus harmonizado
    livre de duplicatas entre ambas as bases.
    """
    tables_dir = output_tables_dir or (HARMONIZED_OUTPUTS_DIR / "tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for label, df in [("Google Patents / USPTO (BigQuery)", df_uspto), ("The Lens (Internacional)", df_lens)]:
        if df.empty:
            summary_rows.append(
                {
                    "Fonte": label,
                    "Famílias Únicas": 0,
                    "Jurisdições Distintas": 0,
                    "Países de Titulares": 0,
                    "Depositantes Únicos": 0,
                    "Período Coberto": "—",
                }
            )
            continue
        summary_rows.append(
            {
                "Fonte": label,
                "Famílias Únicas": int(len(df)),
                "Jurisdições Distintas": int(df["jurisdiction"].nunique()),
                "Países de Titulares": int(df[df["assignee_country"] != "XX"]["assignee_country"].nunique()),
                "Depositantes Únicos": int(df["assignee"].nunique()),
                "Período Coberto": f"{int(df['publication_year'].min())}–{int(df['publication_year'].max())}",
            }
        )

    # Identificação de sobreposição exata (ID normalizado ou hash de título)
    uspto_ids = set(df_uspto["patent_id"].apply(_normalize_id)) if not df_uspto.empty else set()
    lens_ids = set(df_lens["patent_id"].apply(_normalize_id)) if not df_lens.empty else set()
    overlap_ids = uspto_ids.intersection(lens_ids)

    uspto_titles = set(df_uspto["title"].apply(_normalize_title_key)) if not df_uspto.empty else set()
    lens_titles = set(df_lens["title"].apply(_normalize_title_key)) if not df_lens.empty else set()
    overlap_titles = uspto_titles.intersection(lens_titles)

    # Harmonização priorizando os metadados internacionais mais ricos do The Lens
    # e incorporando registros exclusivos do USPTO ODP
    combined = pd.concat([df_lens, df_uspto], ignore_index=True)
    if not combined.empty:
        combined["_norm_id"] = combined["patent_id"].apply(_normalize_id)
        combined["_title_hash"] = combined["title"].apply(_normalize_title_key)
        combined = combined.drop_duplicates(subset=["_norm_id"], keep="first")
        combined = combined.drop_duplicates(subset=["_title_hash"], keep="first")
        combined = combined.drop(columns=["_norm_id", "_title_hash"]).reset_index(drop=True)

    summary_rows.append(
        {
            "Fonte": "Corpus Harmonizado (Unificado)",
            "Famílias Únicas": int(len(combined)),
            "Jurisdições Distintas": int(combined["jurisdiction"].nunique()) if not combined.empty else 0,
            "Países de Titulares": int(combined[combined["assignee_country"] != "XX"]["assignee_country"].nunique()) if not combined.empty else 0,
            "Depositantes Únicos": int(combined["assignee"].nunique()) if not combined.empty else 0,
            "Período Coberto": f"{int(combined['publication_year'].min())}–{int(combined['publication_year'].max())}" if not combined.empty else "—",
        }
    )

    df_summary = pd.DataFrame(summary_rows)
    csv_comp = tables_dir / "comparacao_fontes_uspto_vs_lens.csv"
    df_summary.to_csv(csv_comp, index=False, encoding="utf-8-sig")

    md_comp = tables_dir / "comparacao_fontes_uspto_vs_lens.md"
    with open(md_comp, "w", encoding="utf-8") as f:
        f.write("# Comparação e Compatibilização entre Fontes Patentárias (CPC G06N)\n\n")
        f.write(df_summary.to_markdown(index=False))
        f.write(
            f"\n\n- **Sobreposição por Número de Publicação Normalizado:** {len(overlap_ids)} registros.\n"
            f"- **Sobreposição por Título Normalizado:** {len(overlap_titles)} registros.\n"
        )

    harm_path = PROCESSED_DATA_DIR / "patentes_g06n_harmonizado.csv"
    combined.to_csv(harm_path, index=False, encoding="utf-8-sig")
    logger.info(
        f"Comparação concluída ({len(overlap_ids)} sobreposições por ID; "
        f"{len(combined)} famílias no corpus harmonizado salvo em {harm_path})."
    )
    return df_summary, combined
