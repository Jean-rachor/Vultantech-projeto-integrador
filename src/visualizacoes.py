# ==============================================================================
# VulcanTech Smart Metals S.A. — Análise Exploratória e Visualizações
# ------------------------------------------------------------------------------
# Script completo de EDA com pandas, scipy, seaborn e matplotlib.
# Gera gráficos profissionais organizados por categoria de análise.
# ==============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import statistics as st
import seaborn as srn
import matplotlib
matplotlib.use("Agg")  # Backend não-interativo para salvar figuras
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# Configurações globais de estilo
# ------------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})
srn.set_style("whitegrid")
PALETA_FALHAS = {0: "#2ecc71", 1: "#f39c12", 2: "#3498db", 3: "#e74c3c"}
PALETA_LIGAS = {"Aco_Carbono_316L": "#95a5a6",
                 "Aluminio_7075": "#3498db",
                 "Liga_Titanio_Ti6Al4V": "#e67e22"}
PALETA_CELULAS = ["#1abc9c", "#9b59b6", "#3498db", "#e74c3c"]
ROTULOS_FALHA = {0: "Normal", 1: "Desgaste Ferramenta",
                  2: "Falha Refrigeração", 3: "Desalinhamento/Trinca"}

# ------------------------------------------------------------------------------
# Caminhos
# ------------------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE, "Data", "dados_sensores_vulcantech.csv")
OUT_DIR = os.path.join(BASE, "Docs", "Entregaveis", "Graficos")
os.makedirs(OUT_DIR, exist_ok=True)

COLUNAS_SENSORES = [
    "temperatura_fuso_c", "vibracao_triaxial_rms_mms", "torque_fuso_nm",
    "vazao_refrigerante_lmin", "corrente_estator_amp",
    "pressao_hidraulica_bar", "nivel_ruido_db",
]


def salvar_fig(fig, nome):
    """Salva a figura e fecha-a."""
    caminho = os.path.join(OUT_DIR, nome)
    fig.savefig(caminho)
    plt.close(fig)
    print(f"  [OK] {nome}")


# ==============================================================================
# 1. CARGA E TRATAMENTO DOS DADOS (melhorado em relação ao tratamento.py)
# ==============================================================================
def carregar_e_tratar_dados():
    print("\n[1] Carregando e tratando dados...")

    df = pd.read_csv(CSV_PATH)

    # --- Conversão de timestamp e ordenação ---
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # --- Identificação de ruídos físicos (erros de leitura) ---
    ruido_temp = df["temperatura_fuso_c"] >= 400       # 999.9°C
    ruido_vib  = df["vibracao_triaxial_rms_mms"] < 0   # -99.0 mm/s
    ruido_vaz  = df["vazao_refrigerante_lmin"] < 0
    ruido_pres = df["pressao_hidraulica_bar"] < 0       # -1.0 bar

    print(f"  Ruídos detectados → temp:{ruido_temp.sum()}, "
          f"vib:{ruido_vib.sum()}, vaz:{ruido_vaz.sum()}, pres:{ruido_pres.sum()}")

    # --- Substituir ruídos por NaN ---
    df.loc[ruido_temp, "temperatura_fuso_c"] = np.nan
    df.loc[ruido_vib, "vibracao_triaxial_rms_mms"] = np.nan
    df.loc[ruido_vaz, "vazao_refrigerante_lmin"] = np.nan
    df.loc[ruido_pres, "pressao_hidraulica_bar"] = np.nan

    # --- Interpolação temporal condicional por equipamento (id_celula) ---
    # O PDF exige interpolação temporal por célula, não média global.
    df = df.set_index("timestamp")
    for col in ["temperatura_fuso_c", "vibracao_triaxial_rms_mms",
                "vazao_refrigerante_lmin", "pressao_hidraulica_bar"]:
        # Interpolação temporal (method='time') dentro de cada célula
        df[col] = df.groupby("id_celula", group_keys=False)[col]\
            .apply(lambda g: g.interpolate(method="time"))
        # Fallback: se ainda houver NaN (ex: no início da série), mediana por liga
        if df[col].isnull().any():
            mediana = df.groupby("liga_metalica")[col].transform("median")
            df[col] = df[col].fillna(mediana)
            df[col] = df[col].fillna(df[col].median())  # último recurso

    df = df.reset_index()

    print(f"  Após tratamento — NaNs restantes: {df[COLUNAS_SENSORES].isnull().sum().sum()}")
    return df


# ==============================================================================
# 2. ENGENHARIA DE FEATURES (mantida e aprimorada do tratamento.py)
# ==============================================================================
def engenharia_features(df):
    print("\n[2] Engenharia de features...")
    df = df.sort_values("timestamp").set_index("timestamp")

    # --- Rolling means e desvios (janelas de 30 min e 2 h) ---
    for col in COLUNAS_SENSORES:
        df[f"{col}_media_30min"] = df[col].rolling("30min", min_periods=1).mean()
        df[f"{col}_desvio_30min"] = df[col].rolling("30min", min_periods=1).std()
        df[f"{col}_media_2h"] = df[col].rolling("2h", min_periods=1).mean()
        df[f"{col}_desvio_2h"] = df[col].rolling("2h", min_periods=1).std()

    # --- Índice de Esforço Mecânico (IEM) ---
    # IEM = (Torque × Vibração RMS) / (Vazão + 1)
    df["IEM"] = (df["torque_fuso_nm"] * df["vibracao_triaxial_rms_mms"]) / \
                (df["vazao_refrigerante_lmin"] + 1)

    # --- Normalização pela média histórica da liga metálica ---
    for col in COLUNAS_SENSORES:
        media_hist = df.groupby("liga_metalica")[col].transform("mean")
        df[f"{col}_norm"] = df[col] / media_hist.replace(0, np.nan)

    df = df.reset_index()
    print(f"  Features criadas. Total de colunas: {df.shape[1]}")
    return df


# ==============================================================================
# GRUPO A — Visão Geral e Qualidade dos Dados
# ==============================================================================
def graficos_visao_geral(df):
    print("\n[A] Gerando gráficos de visão geral...")

    # --- A1: Painel de distribuição do alvo (tipo_falha) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    counts = df["tipo_falha"].value_counts().sort_index()
    cores = [PALETA_FALHAS[k] for k in counts.index]

    ax = axes[0]
    bars = ax.bar([ROTULOS_FALHA[k] for k in counts.index],
                  counts.values, color=cores, edgecolor="black", linewidth=0.5)
    ax.set_title("Distribuição das Falhas (Contagem Absoluta)")
    ax.set_ylabel("Nº de Registros")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f"{val}\n({val/len(df)*100:.1f}%)",
                ha="center", va="bottom", fontsize=9)

    ax2 = axes[1]
    ax2.pie(counts.values, labels=[ROTULOS_FALHA[k] for k in counts.index],
            colors=cores, autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax2.set_title("Proporção de Falhas (%)")
    plt.tight_layout()
    salvar_fig(fig, "A1_distribuicao_falhas.png")

    # --- A2: Registros por célula e por liga ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cel_counts = df["id_celula"].value_counts().sort_index()
    axes[0].bar(cel_counts.index, cel_counts.values, color=PALETA_CELULAS,
                edgecolor="black", linewidth=0.5)
    axes[0].set_title("Registros por Célula de Manufatura")
    axes[0].set_ylabel("Nº de Registros")
    for i, v in enumerate(cel_counts.values):
        axes[0].text(i, v + 30, str(v), ha="center", fontsize=9)

    lig_counts = df["liga_metalica"].value_counts()
    axes[1].barh(lig_counts.index, lig_counts.values,
                 color=[PALETA_LIGAS[k] for k in lig_counts.index],
                 edgecolor="black", linewidth=0.5)
    axes[1].set_title("Registros por Liga Metálica")
    axes[1].set_xlabel("Nº de Registros")
    for i, v in enumerate(lig_counts.values):
        axes[1].text(v + 30, i, str(v), va="center", fontsize=9)

    plt.tight_layout()
    salvar_fig(fig, "A2_registros_celula_liga.png")

    # --- A3: Distribuição cruzada Célula × Tipo de Falha ---
    cross = pd.crosstab(df["id_celula"], df["tipo_falha"])
    cross.columns = [ROTULOS_FALHA[c] for c in cross.columns]
    fig, ax = plt.subplots(figsize=(10, 5))
    cross.plot(kind="bar", stacked=True, ax=ax,
               color=[PALETA_FALHAS[k] for k in cross.columns.map(
                   {v: k for k, v in ROTULOS_FALHA.items()})],
               edgecolor="black", linewidth=0.4)
    ax.set_title("Distribuição de Falhas por Célula (Stacked)")
    ax.set_ylabel("Nº de Registros")
    ax.legend(title="Tipo de Falha", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    salvar_fig(fig, "A3_falhas_por_celula_stacked.png")

    # --- A4: Distribuição cruzada Liga × Tipo de Falha ---
    cross2 = pd.crosstab(df["liga_metalica"], df["tipo_falha"])
    cross2.columns = [ROTULOS_FALHA[c] for c in cross2.columns]
    fig, ax = plt.subplots(figsize=(10, 5))
    cross2.plot(kind="bar", stacked=True, ax=ax,
                color=[PALETA_FALHAS[k] for k in cross2.columns.map(
                    {v: k for k, v in ROTULOS_FALHA.items()})],
                edgecolor="black", linewidth=0.4)
    ax.set_title("Distribuição de Falhas por Liga Metálica (Stacked)")
    ax.set_ylabel("Nº de Registros")
    ax.legend(title="Tipo de Falha", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
    plt.tight_layout()
    salvar_fig(fig, "A4_falhas_por_liga_stacked.png")


# ==============================================================================
# GRUPO B — Análise Unidimensional (Histogramas e Boxplots)
# ==============================================================================
def graficos_unidimensionais(df):
    print("\n[B] Gerando histogramas e boxplots unidimensionais...")

    rotulos_curtos = {
        "temperatura_fuso_c": "Temperatura do Fuso (°C)",
        "vibracao_triaxial_rms_mms": "Vibração Triaxial RMS (mm/s)",
        "torque_fuso_nm": "Torque do Fuso (Nm)",
        "vazao_refrigerante_lmin": "Vazão Refrigerante (L/min)",
        "corrente_estator_amp": "Corrente Estator (A)",
        "pressao_hidraulica_bar": "Pressão Hidráulica (bar)",
        "nivel_ruido_db": "Nível de Ruído (dB)",
    }

    # --- B1: Histogramas com KDE de todos os sensores ---
    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    axes = axes.flatten()
    for i, col in enumerate(COLUNAS_SENSORES):
        srn.histplot(df[col], kde=True, ax=axes[i], color="#2c3e50", bins=50)
        axes[i].set_title(f"Distribuição — {rotulos_curtos[col]}")
        axes[i].axvline(df[col].mean(), color="#e74c3c", linestyle="--",
                        label=f"Média: {df[col].mean():.2f}")
        axes[i].axvline(df[col].median(), color="#2ecc71", linestyle="--",
                        label=f"Mediana: {df[col].median():.2f}")
        axes[i].legend(fontsize=8)
    axes[-1].axis("off")
    plt.tight_layout()
    salvar_fig(fig, "B1_histogramas_sensores.png")

    # --- B2: Boxplots de todos os sensores ---
    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    axes = axes.flatten()
    for i, col in enumerate(COLUNAS_SENSORES):
        srn.boxplot(data=df, x=col, ax=axes[i], color="#3498db", fliersize=2)
        axes[i].set_title(f"Boxplot — {rotulos_curtos[col]}")
    axes[-1].axis("off")
    plt.tight_layout()
    salvar_fig(fig, "B2_boxplots_sensores.png")


# ==============================================================================
# GRUPO C — Análise por Falha (sensores vs tipo_falha)
# ==============================================================================
def graficos_por_falha(df):
    print("\n[C] Gerando análise por tipo de falha...")

    rotulos_curtos = {
        "temperatura_fuso_c": "Temperatura do Fuso (°C)",
        "vibracao_triaxial_rms_mms": "Vibração Triaxial RMS (mm/s)",
        "torque_fuso_nm": "Torque do Fuso (Nm)",
        "vazao_refrigerante_lmin": "Vazão Refrigerante (L/min)",
        "corrente_estator_amp": "Corrente Estator (A)",
        "pressao_hidraulica_bar": "Pressão Hidráulica (bar)",
        "nivel_ruido_db": "Nível de Ruído (dB)",
    }

    df_plot = df.copy()
    df_plot["falha_rotulo"] = df_plot["tipo_falha"].map(ROTULOS_FALHA)

    # --- C1: Boxplots de cada sensor por tipo de falha ---
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    axes = axes.flatten()
    for i, col in enumerate(COLUNAS_SENSORES):
        srn.boxplot(data=df_plot, x="falha_rotulo", y=col, ax=axes[i],
                    palette=PALETA_FALHA_PLOT, fliersize=1.5, linewidth=0.8)
        axes[i].set_title(f"{rotulos_curtos[col]} por Tipo de Falha")
        axes[i].set_xlabel("")
    axes[-1].axis("off")
    plt.tight_layout()
    salvar_fig(fig, "C1_boxplots_por_falha.png")

    # --- C2: Violin plots de sensores-chave por falha ---
    sensores_chave = ["temperatura_fuso_c", "vibracao_triaxial_rms_mms",
                      "torque_fuso_nm", "corrente_estator_amp"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    for i, col in enumerate(sensores_chave):
        srn.violinplot(data=df_plot, x="falha_rotulo", y=col, ax=axes[i],
                       palette=PALETA_FALHA_PLOT, inner="quartile", linewidth=0.8)
        axes[i].set_title(f"Violin — {rotulos_curtos[col]} por Falha")
        axes[i].set_xlabel("")
    plt.tight_layout()
    salvar_fig(fig, "C2_violinplots_por_falha.png")

    # --- C3: Média dos sensores por tipo de falha (heatmap) ---
    medias_falha = df_plot.groupby("tipo_falha")[COLUNAS_SENSORES].mean()
    medias_falha.index = [ROTULOS_FALHA[k] for k in medias_falha.index]
    fig, ax = plt.subplots(figsize=(12, 5))
    srn.heatmap(medias_falha, annot=True, fmt=".2f", cmap="YlOrRd",
                ax=ax, linewidths=0.5, cbar_kws={"label": "Valor Médio"})
    ax.set_title("Média dos Sensores por Tipo de Falha")
    ax.set_ylabel("")
    plt.tight_layout()
    salvar_fig(fig, "C3_media_sensores_por_falha_heatmap.png")

    # --- C4: Mediana dos sensores por tipo de falha (heatmap) ---
    medianas_falha = df_plot.groupby("tipo_falha")[COLUNAS_SENSORES].median()
    medianas_falha.index = [ROTULOS_FALHA[k] for k in medianas_falha.index]
    fig, ax = plt.subplots(figsize=(12, 5))
    srn.heatmap(medianas_falha, annot=True, fmt=".2f", cmap="YlGnBu",
                ax=ax, linewidths=0.5, cbar_kws={"label": "Mediana"})
    ax.set_title("Mediana dos Sensores por Tipo de Falha")
    ax.set_ylabel("")
    plt.tight_layout()
    salvar_fig(fig, "C4_mediana_sensores_por_falha_heatmap.png")


# ==============================================================================
# GRUPO D — Análise por Liga Metálica (falsos alarmes)
# ==============================================================================
def graficos_por_liga(df):
    print("\n[D] Gerando análise por liga metálica...")

    rotulos_curtos = {
        "temperatura_fuso_c": "Temperatura (°C)",
        "vibracao_triaxial_rms_mms": "Vibração (mm/s)",
        "torque_fuso_nm": "Torque (Nm)",
        "vazao_refrigerante_lmin": "Vazão (L/min)",
        "corrente_estator_amp": "Corrente (A)",
        "pressao_hidraulica_bar": "Pressão (bar)",
        "nivel_ruido_db": "Ruído (dB)",
    }

    # --- D1: Boxplots de sensores-chave por liga (motivo dos falsos alarmes) ---
    sensores_chave = ["torque_fuso_nm", "corrente_estator_amp",
                      "temperatura_fuso_c", "vibracao_triaxial_rms_mms"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    for i, col in enumerate(sensores_chave):
        srn.boxplot(data=df, x="liga_metalica", y=col, ax=axes[i],
                    palette=PALETA_LIGAS, fliersize=1.5, linewidth=0.8)
        axes[i].set_title(f"{rotulos_curtos[col]} por Liga Metálica")
        axes[i].set_xlabel("")
        axes[i].tick_params(axis="x", rotation=15)
    plt.suptitle("Falsos Alarmes: Ligas duras (Titânio) elevam torque e corrente\n"
                 "em operação normal", fontsize=12, y=1.01)
    plt.tight_layout()
    salvar_fig(fig, "D1_boxplots_por_liga_falsos_alarmes.png")

    # --- D2: Média dos sensores por liga metálica ---
    medias_liga = df.groupby("liga_metalica")[COLUNAS_SENSORES].mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    srn.heatmap(medias_liga, annot=True, fmt=".2f", cmap="coolwarm",
                ax=ax, linewidths=0.5, cbar_kws={"label": "Valor Médio"})
    ax.set_title("Média dos Sensores por Liga Metálica")
    ax.set_ylabel("")
    plt.tight_layout()
    salvar_fig(fig, "D2_media_sensores_por_liga_heatmap.png")

    # --- D3: Violin plots de torque e corrente por liga (falsos alarmes) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    srn.violinplot(data=df, x="liga_metalica", y="torque_fuso_nm", ax=axes[0],
                   palette=PALETA_LIGAS, inner="quartile", linewidth=0.8)
    axes[0].set_title("Torque por Liga Metálica")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=15)

    srn.violinplot(data=df, x="liga_metalica", y="corrente_estator_amp", ax=axes[1],
                   palette=PALETA_LIGAS, inner="quartile", linewidth=0.8)
    axes[1].set_title("Corrente Estator por Liga Metálica")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=15)
    plt.tight_layout()
    salvar_fig(fig, "D3_violin_torque_corrente_por_liga.png")

    # --- D4: Distribuição de falhas por liga (proporção) ---
    cross_liga = pd.crosstab(df["liga_metalica"], df["tipo_falha"], normalize="index") * 100
    cross_liga.columns = [ROTULOS_FALHA[c] for c in cross_liga.columns]
    fig, ax = plt.subplots(figsize=(12, 5))
    cross_liga.plot(kind="bar", stacked=True, ax=ax,
                    color=[PALETA_FALHAS[k] for k in cross_liga.columns.map(
                        {v: k for k, v in ROTULOS_FALHA.items()})],
                    edgecolor="black", linewidth=0.4)
    ax.set_title("Proporção de Falhas por Liga Metálica (%)")
    ax.set_ylabel("Percentual (%)")
    ax.legend(title="Tipo de Falha", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
    plt.tight_layout()
    salvar_fig(fig, "D4_proporcao_falhas_por_liga.png")


# ==============================================================================
# GRUPO E — Correlação e Relações Multivariadas
# ==============================================================================
def graficos_correlacao(df):
    print("\n[E] Gerando análise de correlação...")

    # --- E1: Matriz de correlação (Pearson) ---
    cols_corr = COLUNAS_SENSORES + ["tipo_falha"]
    corr = df[cols_corr].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    srn.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5,
                square=True, cbar_kws={"shrink": 0.8, "label": "Correlação (Pearson)"})
    ax.set_title("Matriz de Correlação — Sensores e Tipo de Falha (Pearson)")
    plt.tight_layout()
    salvar_fig(fig, "E1_correlacao_pearson.png")

    # --- E2: Matriz de correlação (Spearman) ---
    corr_s = df[cols_corr].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(11, 9))
    srn.heatmap(corr_s, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5,
                square=True, cbar_kws={"shrink": 0.8, "label": "Correlação (Spearman)"})
    ax.set_title("Matriz de Correlação — Sensores e Tipo de Falha (Spearman)")
    plt.tight_layout()
    salvar_fig(fig, "E2_correlacao_spearman.png")

    # --- E3: Pairplot dos sensores-chave colorido por falha ---
    df_plot = df.copy()
    df_plot["falha_rotulo"] = df_plot["tipo_falha"].map(ROTULOS_FALHA)
    pair_cols = ["temperatura_fuso_c", "vibracao_triaxial_rms_mms",
                 "torque_fuso_nm", "corrente_estator_amp", "falha_rotulo"]
    # Sample para acelerar
    df_sample = df_plot[pair_cols].sample(min(3000, len(df_plot)), random_state=42)
    g = srn.pairplot(df_sample, hue="falha_rotulo",
                     palette=PALETA_FALHA_PLOT,
                     diag_kind="kde", plot_kws={"s": 12, "alpha": 0.6},
                     corner=True)
    g.figure.suptitle("Pairplot — Sensores-Chave por Tipo de Falha",
                      y=1.02, fontsize=14, fontweight="bold")
    salvar_fig(g.figure, "E3_pairplot_sensores_falha.png")

    # --- E4: Scatter torque × corrente colorido por liga ---
    fig, ax = plt.subplots(figsize=(10, 8))
    for liga, cor in PALETA_LIGAS.items():
        subset = df[df["liga_metalica"] == liga]
        ax.scatter(subset["torque_fuso_nm"], subset["corrente_estator_amp"],
                   c=cor, label=liga, alpha=0.3, s=10, edgecolors="none")
    ax.set_xlabel("Torque do Fuso (Nm)")
    ax.set_ylabel("Corrente do Estator (A)")
    ax.set_title("Torque × Corrente por Liga Metálica\n"
                 "(Evidência de falsos alarmes: Titânio ocupa faixas mais altas)")
    ax.legend(markerscale=3)
    plt.tight_layout()
    salvar_fig(fig, "E4_scatter_torque_corrente_liga.png")

    # --- E5: Scatter temperatura × vibração colorido por falha ---
    fig, ax = plt.subplots(figsize=(10, 8))
    for cod, rotulo in ROTULOS_FALHA.items():
        subset = df[df["tipo_falha"] == cod]
        ax.scatter(subset["temperatura_fuso_c"], subset["vibracao_triaxial_rms_mms"],
                   c=PALETA_FALHAS[cod], label=rotulo, alpha=0.3, s=10,
                   edgecolors="none")
    ax.set_xlabel("Temperatura do Fuso (°C)")
    ax.set_ylabel("Vibração Triaxial RMS (mm/s)")
    ax.set_title("Temperatura × Vibração por Tipo de Falha")
    ax.legend(markerscale=3)
    plt.tight_layout()
    salvar_fig(fig, "E5_scatter_temp_vib_falha.png")


# ==============================================================================
# GRUPO F — Séries Temporais
# ==============================================================================
def graficos_series_temporais(df):
    print("\n[F] Gerando séries temporais...")

    rotulos_curtos = {
        "temperatura_fuso_c": "Temperatura (°C)",
        "vibracao_triaxial_rms_mms": "Vibração (mm/s)",
        "torque_fuso_nm": "Torque (Nm)",
        "vazao_refrigerante_lmin": "Vazão (L/min)",
        "corrente_estator_amp": "Corrente (A)",
        "pressao_hidraulica_bar": "Pressão (bar)",
        "nivel_ruido_db": "Ruído (dB)",
    }

    # --- F1: Série temporal de todos os sensores (amostrada por hora) ---
    df_ts = df.set_index("timestamp")
    df_hourly = df_ts[COLUNAS_SENSORES].resample("1h").mean()

    fig, axes = plt.subplots(7, 1, figsize=(16, 22), sharex=True)
    for i, col in enumerate(COLUNAS_SENSORES):
        axes[i].plot(df_hourly.index, df_hourly[col], color="#2c3e50",
                     linewidth=0.8, alpha=0.8)
        axes[i].fill_between(df_hourly.index, df_hourly[col],
                            alpha=0.15, color="#2c3e50")
        axes[i].set_ylabel(rotulos_curtos[col])
        axes[i].grid(True, alpha=0.3)
    axes[0].set_title("Séries Temporais — Média Horária dos Sensores")
    axes[-1].set_xlabel("Data/Hora")
    plt.tight_layout()
    salvar_fig(fig, "F1_series_temporais_sensores.png")

    # --- F2: Série temporal de temperatura com eventos de falha ---
    fig, ax = plt.subplots(figsize=(16, 6))
    # Plotar apenas uma célula para clareza
    df_cel = df[df["id_celula"] == "CEL-201"].set_index("timestamp")
    ax.plot(df_cel.index, df_cel["temperatura_fuso_c"], color="#2c3e50",
            linewidth=0.5, alpha=0.7, label="Temperatura")

    # Marcar falhas
    for cod, rotulo in ROTULOS_FALHA.items():
        if cod == 0:
            continue
        mask = df_cel["tipo_falha"] == cod
        ax.scatter(df_cel.index[mask], df_cel["temperatura_fuso_c"][mask],
                   c=PALETA_FALHAS[cod], s=20, label=rotulo, zorder=5, alpha=0.8)
    ax.set_ylabel("Temperatura do Fuso (°C)")
    ax.set_xlabel("Data/Hora")
    ax.set_title("Série Temporal — Temperatura (CEL-201) com Eventos de Falha")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    salvar_fig(fig, "F2_temp_temperatura_falhas_cel201.png")

    # --- F3: Rolling means de temperatura por liga ---
    fig, ax = plt.subplots(figsize=(16, 6))
    for liga, cor in PALETA_LIGAS.items():
        df_liga = df[df["liga_metalica"] == liga].set_index("timestamp")
        rolling = df_liga["temperatura_fuso_c"].rolling("2h", min_periods=1).mean()
        ax.plot(rolling.index, rolling.values, color=cor, linewidth=0.8,
                alpha=0.7, label=liga)
    ax.set_ylabel("Temperatura — Média Móvel 2h (°C)")
    ax.set_xlabel("Data/Hora")
    ax.set_title("Média Móvel (2h) da Temperatura por Liga Metálica")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    salvar_fig(fig, "F3_rolling_temp_por_liga.png")

    # --- F4: Rolling std (desvio) de vibração por liga ---
    fig, ax = plt.subplots(figsize=(16, 6))
    for liga, cor in PALETA_LIGAS.items():
        df_liga = df[df["liga_metalica"] == liga].set_index("timestamp")
        rolling_std = df_liga["vibracao_triaxial_rms_mms"].rolling("2h", min_periods=1).std()
        ax.plot(rolling_std.index, rolling_std.values, color=cor, linewidth=0.8,
                alpha=0.7, label=liga)
    ax.set_ylabel("Desvio Padrão Móvel 2h — Vibração (mm/s)")
    ax.set_xlabel("Data/Hora")
    ax.set_title("Variabilidade (Rolling Std 2h) da Vibração por Liga Metálica")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    salvar_fig(fig, "F4_rolling_std_vib_por_liga.png")

    # --- F5: Eventos de falha ao longo do tempo (timeline) ---
    fig, ax = plt.subplots(figsize=(16, 5))
    df_falhas = df[df["tipo_falha"] != 0].copy()
    for cod, rotulo in ROTULOS_FALHA.items():
        if cod == 0:
            continue
        subset = df_falhas[df_falhas["tipo_falha"] == cod]
        ax.scatter(subset["timestamp"], [cod] * len(subset),
                   c=PALETA_FALHAS[cod], s=8, alpha=0.5, label=rotulo)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels([ROTULOS_FALHA[1], ROTULOS_FALHA[2], ROTULOS_FALHA[3]])
    ax.set_xlabel("Data/Hora")
    ax.set_title("Linha do Tempo — Eventos de Falha")
    ax.legend(markerscale=4, loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    salvar_fig(fig, "F5_timeline_falhas.png")


# ==============================================================================
# GRUPO G — Feature Engineering: IEM e Normalização
# ==============================================================================
def graficos_features(df):
    print("\n[G] Gerando gráficos de features avançadas...")

    df_plot = df.copy()
    df_plot["falha_rotulo"] = df_plot["tipo_falha"].map(ROTULOS_FALHA)

    # --- G1: Distribuição do IEM por tipo de falha ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    srn.histplot(data=df_plot, x="IEM", hue="falha_rotulo", kde=True,
                 ax=axes[0], palette=PALETA_FALHA_PLOT, bins=60, alpha=0.5)
    axes[0].set_title("Distribuição do IEM por Tipo de Falha")
    axes[0].set_xlabel("Índice de Esforço Mecânico (IEM)")
    axes[0].set_xlim(0, df_plot["IEM"].quantile(0.99))

    srn.boxplot(data=df_plot, x="falha_rotulo", y="IEM", ax=axes[1],
                palette=PALETA_FALHA_PLOT, fliersize=1.5)
    axes[1].set_title("IEM por Tipo de Falha")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("IEM")
    plt.tight_layout()
    salvar_fig(fig, "G1_distribuicao_IEM.png")

    # --- G2: Série temporal do IEM ---
    fig, ax = plt.subplots(figsize=(16, 5))
    df_ts = df.set_index("timestamp")
    df_hourly = df_ts["IEM"].resample("1h").mean()
    ax.plot(df_hourly.index, df_hourly.values, color="#8e44ad", linewidth=0.8)
    ax.fill_between(df_hourly.index, df_hourly.values, alpha=0.15, color="#8e44ad")
    ax.set_ylabel("IEM (média horária)")
    ax.set_xlabel("Data/Hora")
    ax.set_title("Série Temporal — Índice de Esforço Mecânico (IEM)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    salvar_fig(fig, "G2_serie_temporal_IEM.png")

    # --- G3: Sensores normalizados por liga (boxplots) ---
    cols_norm = ["torque_fuso_nm_norm", "corrente_estator_amp_norm",
                 "temperatura_fuso_c_norm", "vibracao_triaxial_rms_mms_norm"]
    rotulos_norm = ["Torque (norm.)", "Corrente (norm.)",
                    "Temperatura (norm.)", "Vibração (norm.)"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, (col, rot) in enumerate(zip(cols_norm, rotulos_norm)):
        srn.boxplot(data=df_plot, x="liga_metalica", y=col, ax=axes[i],
                    palette=PALETA_LIGAS, fliersize=1.5, linewidth=0.8)
        axes[i].axhline(1.0, color="#e74c3c", linestyle="--", linewidth=1,
                        label="Média histórica (1.0)")
        axes[i].set_title(f"{rot} — Normalizado por Liga")
        axes[i].set_xlabel("")
        axes[i].tick_params(axis="x", rotation=15)
        axes[i].legend(fontsize=8)
    plt.tight_layout()
    salvar_fig(fig, "G3_sensores_normalizados_por_liga.png")

    # --- G4: Matriz de correlação com features derivadas ---
    cols_deriv = (["temperatura_fuso_c", "vibracao_triaxial_rms_mms",
                   "torque_fuso_nm", "vazao_refrigerante_lmin",
                   "corrente_estator_amp", "pressao_hidraulica_bar",
                   "nivel_ruido_db", "IEM",
                   "temperatura_fuso_c_media_30min",
                   "vibracao_triaxial_rms_mms_media_30min",
                   "torque_fuso_nm_desvio_2h", "tipo_falha"])
    corr_deriv = df_plot[cols_deriv].corr()
    fig, ax = plt.subplots(figsize=(13, 11))
    mask = np.triu(np.ones_like(corr_deriv, dtype=bool), k=1)
    srn.heatmap(corr_deriv, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5,
                square=True, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlação — Sensores + Features Derivadas + IEM + Falha")
    plt.tight_layout()
    salvar_fig(fig, "G4_correlacao_features_derivadas.png")


# ==============================================================================
# GRUPO H — OEE (Overall Equipment Effectiveness) Estimado
# ==============================================================================
def graficos_oee(df):
    print("\n[H] Gerando gráficos de OEE...")

    # OEE = Disponibilidade × Performance × Qualidade
    # Estimativa simplificada:
    # Disponibilidade = % de tempo sem falha
    # Performance = baseada na média de torque/vazão (proxy de velocidade)
    # Qualidade = inverso de falhas de desgaste/desalinhamento

    df_oee = df.copy()
    df_oee["timestamp"] = pd.to_datetime(df_oee["timestamp"])

    # Cálculo por hora e por célula
    df_oee["hora"] = df_oee["timestamp"].dt.floor("h")
    oee_hora = df_oee.groupby(["id_celula", "hora"]).agg(
        disp=("tipo_falha", lambda x: (x == 0).mean()),
        n_falhas=("tipo_falha", lambda x: (x != 0).sum()),
        torque_med=("torque_fuso_nm", "mean"),
        vazao_med=("vazao_refrigerante_lmin", "mean"),
    ).reset_index()

    # Performance: proxy = razão entre vazão média e vazão máxima esperada (40 L/min)
    oee_hora["performance"] = (oee_hora["vazao_med"] / 40.0).clip(0, 1)
    # Qualidade: proxy = 1 - (falhas de desgaste + desalinhamento) / total
    total_por_hora = df_oee.groupby(["id_celula", "hora"]).size()
    falhas_qualidade = df_oee[df_oee["tipo_falha"].isin([1, 3])].groupby(
        ["id_celula", "hora"]).size()
    oee_hora = oee_hora.set_index(["id_celula", "hora"])
    oee_hora["qualidade"] = 1 - (falhas_qualidade / total_por_hora).fillna(0)
    oee_hora = oee_hora.reset_index()

    # OEE = Disponibilidade × Performance × Qualidade
    oee_hora["OEE"] = oee_hora["disp"] * oee_hora["performance"] * oee_hora["qualidade"]

    # --- H1: OEE médio por célula ---
    oee_celula = oee_hora.groupby("id_celula")["OEE"].mean() * 100
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(oee_celula.index, oee_celula.values, color=PALETA_CELULAS,
                  edgecolor="black", linewidth=0.5)
    ax.axhline(85, color="#2ecc71", linestyle="--", linewidth=1.5,
               label="Meta Mundial (85%)")
    ax.axhline(60, color="#e74c3c", linestyle="--", linewidth=1.5,
               label="Média Típica (60%)")
    for bar, val in zip(bars, oee_celula.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_title("OEE Médio por Célula de Manufatura")
    ax.set_ylabel("OEE (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    plt.tight_layout()
    salvar_fig(fig, "H1_oee_por_celula.png")

    # --- H2: Componentes do OEE por célula (barra agrupada) ---
    componentes = oee_hora.groupby("id_celula").agg(
        Disponibilidade=("disp", "mean"),
        Performance=("performance", "mean"),
        Qualidade=("qualidade", "mean"),
    ) * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(componentes.index))
    width = 0.25
    ax.bar(x - width, componentes["Disponibilidade"], width,
           label="Disponibilidade", color="#1abc9c", edgecolor="black", linewidth=0.4)
    ax.bar(x, componentes["Performance"], width,
           label="Performance", color="#3498db", edgecolor="black", linewidth=0.4)
    ax.bar(x + width, componentes["Qualidade"], width,
           label="Qualidade", color="#9b59b6", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(componentes.index)
    ax.set_ylabel("Percentual (%)")
    ax.set_title("Componentes do OEE por Célula de Manufatura")
    ax.legend()
    ax.set_ylim(0, 110)
    for i in range(len(componentes)):
        ax.text(i - width, componentes["Disponibilidade"].iloc[i] + 0.5,
                f"{componentes['Disponibilidade'].iloc[i]:.1f}%", ha="center", fontsize=8)
        ax.text(i, componentes["Performance"].iloc[i] + 0.5,
                f"{componentes['Performance'].iloc[i]:.1f}%", ha="center", fontsize=8)
        ax.text(i + width, componentes["Qualidade"].iloc[i] + 0.5,
                f"{componentes['Qualidade'].iloc[i]:.1f}%", ha="center", fontsize=8)
    plt.tight_layout()
    salvar_fig(fig, "H2_componentes_oee.png")

    # --- H3: Série temporal do OEE por célula ---
    fig, ax = plt.subplots(figsize=(16, 6))
    for i, cel in enumerate(sorted(oee_hora["id_celula"].unique())):
        subset = oee_hora[oee_hora["id_celula"] == cel]
        ax.plot(subset["hora"], subset["OEE"] * 100,
                color=PALETA_CELULAS[i], linewidth=0.6, alpha=0.6, label=cel)
    ax.axhline(85, color="#2ecc71", linestyle="--", linewidth=1.5,
               label="Meta Mundial (85%)")
    ax.axhline(60, color="#e74c3c", linestyle="--", linewidth=1.5,
               label="Média Típica (60%)")
    ax.set_ylabel("OEE (%)")
    ax.set_xlabel("Data/Hora")
    ax.set_title("Série Temporal do OEE por Célula (agregado por hora)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    salvar_fig(fig, "H3_serie_temporal_oee.png")

    # --- H4: Heatmap OEE por célula e por dia ---
    oee_hora["data"] = oee_hora["hora"].dt.date
    oee_dia = oee_hora.groupby(["id_celula", "data"])["OEE"].mean().unstack(level=0) * 100
    fig, ax = plt.subplots(figsize=(14, 5))
    srn.heatmap(oee_dia.T, annot=True, fmt=".1f", cmap="RdYlGn",
                ax=ax, linewidths=0.5, vmin=0, vmax=100,
                cbar_kws={"label": "OEE (%)"})
    ax.set_title("Heatmap — OEE Diário por Célula de Manufatura")
    ax.set_xlabel("Data")
    ax.set_ylabel("Célula")
    plt.tight_layout()
    salvar_fig(fig, "H4_heatmap_oee_diario.png")


# ==============================================================================
# GRUPO I — Estatísticas Resumidas
# ==============================================================================
def graficos_estatisticas(df):
    print("\n[I] Gerando gráficos estatísticos...")

    # --- I1: Tabela de estatísticas descritivas (visual) ---
    stats_df = df[COLUNAS_SENSORES].describe().round(2)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")
    tabela = ax.table(cellText=stats_df.values,
                      rowLabels=stats_df.index,
                      colLabels=stats_df.columns,
                      cellLoc="center", loc="center",
                      colWidths=[0.12] * len(stats_df.columns))
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(9)
    tabela.scale(1, 1.8)
    # Colorir cabeçalho
    for j in range(len(stats_df.columns)):
        tabela[0, j].set_facecolor("#2c3e50")
        tabela[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Estatísticas Descritivas — Sensores (dados tratados)",
                 fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    salvar_fig(fig, "I1_estatisticas_descritivas.png")

    # --- I2: Assimetria e Curtose por sensor ---
    fig, ax = plt.subplots(figsize=(12, 6))
    skew_vals = [stats.skew(df[col].dropna()) for col in COLUNAS_SENSORES]
    kurt_vals = [stats.kurtosis(df[col].dropna()) for col in COLUNAS_SENSORES]
    x = np.arange(len(COLUNAS_SENSORES))
    width = 0.35
    ax.bar(x - width/2, skew_vals, width, label="Assimetria (Skew)",
           color="#3498db", edgecolor="black", linewidth=0.4)
    ax.bar(x + width/2, kurt_vals, width, label="Curtose (Kurtosis)",
           color="#e74c3c", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in COLUNAS_SENSORES],
                       fontsize=8, rotation=0)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Assimetria e Curtose dos Sensores")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    salvar_fig(fig, "I2_assimetria_curtose.png")

    # --- I3: Teste de normalidade (Q-Q plots) ---
    sensores_qq = ["temperatura_fuso_c", "vibracao_triaxial_rms_mms",
                   "torque_fuso_nm", "corrente_estator_amp"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, col in enumerate(sensores_qq):
        dados = df[col].dropna().sample(min(2000, len(df)), random_state=42)
        stats.probplot(dados, dist="norm", plot=axes[i])
        axes[i].set_title(f"Q-Q Plot — {col}")
        axes[i].get_lines()[0].set_markerfacecolor("#3498db")
        axes[i].get_lines()[0].set_markersize(4)
        axes[i].get_lines()[1].set_color("#e74c3c")
    plt.suptitle("Teste de Normalidade — Q-Q Plots", fontsize=14, y=1.01)
    plt.tight_layout()
    salvar_fig(fig, "I3_qq_plots_normalidade.png")


# ==============================================================================
# FUNÇÃO PRINCIPAL
# ==============================================================================
def main():
    print("=" * 70)
    print("  VULCANTECH — Análise Exploratória e Visualizações")
    print("=" * 70)

    # 1. Carregar e tratar dados
    df = carregar_e_tratar_dados()

    # 2. Engenharia de features
    df = engenharia_features(df)

    # Paleta para boxplots/violins com falhas
    global PALETA_FALHA_PLOT
    PALETA_FALHA_PLOT = [PALETA_FALHAS[k] for k in sorted(PALETA_FALHAS.keys())]

    # 3. Gerar todos os grupos de gráficos
    graficos_visao_geral(df)
    graficos_unidimensionais(df)
    graficos_por_falha(df)
    graficos_por_liga(df)
    graficos_correlacao(df)
    graficos_series_temporais(df)
    graficos_features(df)
    graficos_oee(df)
    graficos_estatisticas(df)

    print("\n" + "=" * 70)
    print(f"  Concluído! {len(os.listdir(OUT_DIR))} gráficos salvos em:")
    print(f"  {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
