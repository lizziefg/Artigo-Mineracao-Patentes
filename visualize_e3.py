"""
Módulo de Visualizações Científicas para a Sprint E3 (Figuras 300 DPI + Mapas).
Gera:
1. Gráfico de barras horizontal por cluster (Volume n e Participação %).
2. Gráfico de linhas da evolução temporal dos clusters (2019–2026, sem coluna Total).
3. Mapa coroplético mundial da distribuição geográfica dos depósitos/titulares.
"""

import logging
from pathlib import Path
from typing import Dict
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

from config import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Mapeamento básico ISO Alpha-2 -> ISO Alpha-3 para visualização geográfica
ISO2_TO_ISO3: Dict[str, str] = {
    "US": "USA", "CN": "CHN", "JP": "JPN", "KR": "KOR", "DE": "DEU",
    "GB": "GBR", "UK": "GBR", "FR": "FRA", "CA": "CAN", "IN": "IND",
    "IL": "ISR", "TW": "TWN", "CH": "CHE", "NL": "NLD", "SE": "SWE",
    "AU": "AUS", "SG": "SGP", "FI": "FIN", "IT": "ITA", "ES": "ESP",
    "BR": "BRA", "IE": "IRL", "BE": "BEL", "AT": "AUT", "DK": "DNK",
    "NO": "NOR", "RU": "RUS", "PL": "POL", "HK": "HKG", "NZ": "NZL",
}


def plot_cluster_distribution_bar(
    df_docs: pd.DataFrame,
    topic_labels: Dict[int, str],
    output_path: Path,
) -> Path:
    """Gera gráfico de barras complementar à Tabela 2 (ClusterID, Volume n e Participação %)."""
    valid = df_docs[df_docs["bertopic_topic"] != -1].copy()
    counts = valid["bertopic_topic"].value_counts().sort_values(ascending=True)
    total = counts.sum()

    labels = [f"C{tid}: {topic_labels.get(tid, f'Tópico {tid}')}" for tid in counts.index]
    values = counts.values
    pcts = [(v / total) * 100.0 if total > 0 else 0.0 for v in values]

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    bars = ax.barh(labels, values, color="#1f4e78", edgecolor="#0f2c46", height=0.65)

    for bar, val, pct in zip(bars, values, pcts):
        ax.text(
            bar.get_width() + max(values) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"n = {val} ({pct:.1f}%)",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
            color="#1a1a1a",
        )

    ax.set_xlim(0, max(values) * 1.22 if len(values) > 0 else 10)
    ax.set_xlabel("Volume de Famílias Patentárias (n)", fontsize=10, fontweight="bold")
    ax.set_title(
        "Distribuição Amostral por Cluster Temático de IA — CPC G06N (BERTopic)",
        fontsize=11.5,
        fontweight="bold",
        pad=12,
    )
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Gráfico de barras salvo em: {output_path}")
    return output_path


def plot_temporal_evolution_lines(
    df_docs: pd.DataFrame,
    topic_labels: Dict[int, str],
    output_path: Path,
) -> Path:
    """Gera gráfico de linhas complementar à Tabela 3 (evolução 2019–2026 sem coluna Total)."""
    valid = df_docs[df_docs["bertopic_topic"] != -1].copy()
    ct = pd.crosstab(valid["publication_year"], valid["bertopic_topic"]).sort_index()

    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=300)
    palette = plt.cm.tab10.colors

    for idx, tid in enumerate(sorted(ct.columns)):
        label = f"C{tid}: {topic_labels.get(tid, f'Tópico {tid}')}"
        ax.plot(
            ct.index,
            ct[tid],
            marker="o",
            linewidth=2.0,
            markersize=5.5,
            color=palette[idx % len(palette)],
            label=label,
        )

    ax.set_xlabel("Ano de Publicação", fontsize=10, fontweight="bold")
    ax.set_ylabel("Volume Anual de Famílias Patentárias (n)", fontsize=10, fontweight="bold")
    ax.set_title(
        "Evolução Temporal dos Clusters Tecnológicos em IA — CPC G06N (2019–2026)",
        fontsize=11.5,
        fontweight="bold",
        pad=12,
    )
    ax.set_xticks(sorted(ct.index.unique()))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        frameon=False,
        fontsize=8.5,
    )
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Gráfico de linhas temporal salvo em: {output_path}")
    return output_path


def plot_country_choropleth_map(
    df_docs: pd.DataFrame,
    figures_dir: Path,
    country_col: str = "assignee_country",
) -> Path:
    """
    Gera mapa coroplético mundial interativo (HTML) e exporta tabela de concentração por país.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    valid = df_docs[df_docs[country_col].notna() & (~df_docs[country_col].isin(["", "XX"]))].copy()
    if valid.empty:
        logger.warning("Sem códigos válidos de país para geração do mapa coroplético.")
        return figures_dir / "mapa_paises_vazio.html"

    counts = valid[country_col].value_counts().reset_index()
    counts.columns = ["iso2", "count"]
    total = counts["count"].sum()
    counts["pct"] = (counts["count"] / total) * 100.0
    counts["iso3"] = counts["iso2"].map(lambda c: ISO2_TO_ISO3.get(str(c).upper(), str(c).upper()))

    fig = px.choropleth(
        counts,
        locations="iso3",
        color="pct",
        hover_name="iso2",
        hover_data={"count": True, "pct": ":.2f", "iso3": False},
        color_continuous_scale="Blues",
        title="Distribuição Geográfica da Amostra de Patentes por País de Origem (CPC G06N)",
        labels={"pct": "Participação (%)", "count": "Famílias Patentárias"},
    )
    fig.update_layout(geo=dict(showframe=False, showcoastlines=True), height=550)

    html_path = figures_dir / "mapa_participacao_paises.html"
    fig.write_html(str(html_path))
    logger.info(f"Mapa coroplético interativo salvo em: {html_path}")

    # Mapa estático 300 DPI em PNG para inserção direta no manuscrito
    geojson_path = DATA_DIR / "ne_110m_countries.geojson"
    if geojson_path.exists():
        world = gpd.read_file(geojson_path)
        world = world[world["ISO_A3_EH"] != "ATA"].copy()
        merged = world.merge(counts, left_on="ISO_A3_EH", right_on="iso3", how="left")
        merged["pct_plot"] = merged["pct"].fillna(0.0)

        fig_static, ax = plt.subplots(figsize=(12, 6.2), dpi=300)
        max_pct = float(merged["pct_plot"].max()) if len(merged) > 0 else 100.0
        norm = mcolors.PowerNorm(gamma=0.4, vmin=0.0, vmax=max(max_pct, 1.0))

        merged[merged["pct_plot"] == 0].plot(
            color="#f2f2f2",
            linewidth=0.35,
            edgecolor="#b0b0b0",
            ax=ax,
        )
        merged[merged["pct_plot"] > 0].plot(
            column="pct_plot",
            cmap="Blues",
            norm=norm,
            linewidth=0.4,
            edgecolor="#707070",
            legend=True,
            legend_kwds={
                "label": "Participação Amostral (%) — Escala Não Linear (γ = 0,4)",
                "orientation": "horizontal",
                "shrink": 0.6,
                "pad": 0.04,
            },
            ax=ax,
        )
        ax.set_axis_off()
        ax.set_title(
            "Distribuição Geográfica por País de Origem / Jurisdição — CPC G06N (2019–2026)",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        png_path = figures_dir / "fig_mapa_paises.png"
        fig_static.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig_static)
        logger.info(f"Mapa coroplético estático (300 DPI) salvo em: {png_path}")

    return html_path
