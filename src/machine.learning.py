# ==============================================================================
# VulcanTech Smart Metals S.A.
# Etapa 2: Inteligência Preditiva Multiclasse, Análise de OEE e MLOps
# ==============================================================================
# Variável alvo: tipo_falha
#   0 = Operação Normal
#   1 = Desgaste Severo de Ferramenta / Fuso
#   2 = Obstrução no Sistema de Refrigeração / Superaquecimento
#   3 = Desalinhamento da Matriz de Prensa / Trinca Mecânica
# ==============================================================================

# Machine Learning
import sklearn
import xgboost as xgb
import lightgbm as lgb
import imblearn

# Modelos e ferramentas específicas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

# Demais imports
import os
import warnings
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    recall_score, f1_score, roc_auc_score
)

warnings.filterwarnings("ignore")

# Versões
print("=" * 65)
print("  VulcanTech — Machine Learning Pipeline (Etapa 2)")
print("=" * 65)
print(f"  sklearn    : {sklearn.__version__}")
print(f"  xgboost    : {xgb.__version__}")
print(f"  lightgbm   : {lgb.__version__}")
print(f"  imbalanced : {imblearn.__version__}")
print(f"  shap       : {shap.__version__}")

# ------------------------------------------------------------------------------
# Caminhos
# ------------------------------------------------------------------------------
BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE, "Data", "dados_sensores_vulcantech.csv")
OUT_DIR  = os.path.join(BASE, "Docs", "Entregaveis", "ML")
os.makedirs(OUT_DIR, exist_ok=True)

ROTULOS_FALHA = {
    0: "Normal",
    1: "Desgaste Ferramenta",
    2: "Falha Refrigeração",
    3: "Desalinhamento/Trinca",
}
PALETA = ["#2ecc71", "#f39c12", "#3498db", "#e74c3c"]


# ==============================================================================
# 1. CARGA E FEATURE ENGINEERING (reutilizando tratamento da Etapa 1)
# ==============================================================================
def carregar_dados():
    print("\n[1] Carregando e preparando dados...")
    df = pd.read_csv(CSV_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ---- Correção de ruídos físicos ----------------------------------------
    df.loc[df["temperatura_fuso_c"] >= 400,       "temperatura_fuso_c"]        = np.nan
    df.loc[df["vibracao_triaxial_rms_mms"] < 0,   "vibracao_triaxial_rms_mms"] = np.nan
    df.loc[df["vazao_refrigerante_lmin"] < 0,     "vazao_refrigerante_lmin"]   = np.nan
    df.loc[df["pressao_hidraulica_bar"] < 0,      "pressao_hidraulica_bar"]    = np.nan

    # ---- Interpolação temporal por célula ----------------------------------
    df = df.set_index("timestamp")
    for col in ["temperatura_fuso_c", "vibracao_triaxial_rms_mms",
                "vazao_refrigerante_lmin", "pressao_hidraulica_bar"]:
        df[col] = df.groupby("id_celula", group_keys=False)[col].apply(
            lambda g: g.interpolate(method="time"))
        if df[col].isnull().any():
            df[col] = df[col].fillna(
                df.groupby("liga_metalica")[col].transform("median"))
            df[col] = df[col].fillna(df[col].median())
    df = df.reset_index()

    # ---- Engenharia de features -------------------------------------------
    SENSORES = ["temperatura_fuso_c", "vibracao_triaxial_rms_mms",
                "torque_fuso_nm", "vazao_refrigerante_lmin",
                "corrente_estator_amp", "pressao_hidraulica_bar",
                "nivel_ruido_db"]

    df = df.set_index("timestamp")
    for col in SENSORES:
        df[f"{col}_media_30min"]  = df[col].rolling("30min", min_periods=1).mean()
        df[f"{col}_desvio_30min"] = df[col].rolling("30min", min_periods=1).std().fillna(0)
        df[f"{col}_media_2h"]     = df[col].rolling("2h", min_periods=1).mean()
        df[f"{col}_desvio_2h"]    = df[col].rolling("2h", min_periods=1).std().fillna(0)

    # IEM — Índice de Esforço Mecânico
    df["IEM"] = (df["torque_fuso_nm"] * df["vibracao_triaxial_rms_mms"]) / \
                (df["vazao_refrigerante_lmin"] + 1)

    # Normalização pela média histórica da liga metálica
    for col in SENSORES:
        media_hist = df.groupby("liga_metalica")[col].transform("mean")
        df[f"{col}_norm"] = df[col] / media_hist.replace(0, np.nan)

    df = df.reset_index()

    # ---- Encoding categórico ----------------------------------------------
    df["id_celula_enc"]    = df["id_celula"].astype("category").cat.codes
    df["liga_metalica_enc"] = df["liga_metalica"].astype("category").cat.codes

    print(f"  Shape final: {df.shape} | Falhas: {df['tipo_falha'].value_counts().to_dict()}")
    return df


# ==============================================================================
# 2. DIVISÃO TEMPORAL TREINO / TESTE (70% / 30%)
# ==============================================================================
def dividir_temporal(df, features):
    """
    Divisão respeitando a ordem temporal — os primeiros 70% ficam
    no treino e os 30% mais recentes formam o teste.
    Evita vazamento de dados temporais (data leakage).
    """
    print("\n[2] Divisão temporal Treino/Teste (70/30)...")
    df_ord = df.sort_values("timestamp").reset_index(drop=True)
    corte  = int(len(df_ord) * 0.70)

    X_train = df_ord.iloc[:corte][features]
    X_test  = df_ord.iloc[corte:][features]
    y_train = df_ord.iloc[:corte]["tipo_falha"]
    y_test  = df_ord.iloc[corte:]["tipo_falha"]

    print(f"  Treino: {len(X_train):,} amostras | Teste: {len(X_test):,} amostras")
    print(f"  Dist. treino: {dict(y_train.value_counts().sort_index())}")
    print(f"  Dist. teste : {dict(y_test.value_counts().sort_index())}")
    return X_train, X_test, y_train, y_test


# ==============================================================================
# 3. TRATAMENTO DO DESBALANCEAMENTO COM SMOTE
# ==============================================================================
def aplicar_smote(X_train, y_train):
    """
    SMOTE — Synthetic Minority Over-sampling Technique.
    Gera amostras sintéticas para as classes minoritárias (1, 2, 3)
    para equilibrar o dataset de treino.
    """
    print("\n[3] Aplicando SMOTE para balanceamento das classes...")
    print(f"  Antes do SMOTE : {dict(pd.Series(y_train).value_counts().sort_index())}")

    smote = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print(f"  Após SMOTE    : {dict(pd.Series(y_res).value_counts().sort_index())}")
    print(f"  Amostras adicionadas: {len(X_res) - len(X_train):,}")
    return X_res, y_res


# ==============================================================================
# 4. NORMALIZAÇÃO (StandardScaler)
# ==============================================================================
def normalizar(X_train, X_test):
    print("\n[4] Normalizando features com StandardScaler...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    return X_train_sc, X_test_sc, scaler


# ==============================================================================
# 5. TREINO DOS MODELOS
# ==============================================================================
def treinar_modelos(X_train, y_train):
    print("\n[5] Treinando modelos ensemble multiclasse...")

    modelos = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            class_weight="balanced",   # peso adicional p/ classes raras
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
    }

    treinados = {}
    for nome, modelo in modelos.items():
        print(f"  Treinando {nome}...")
        modelo.fit(X_train, y_train)
        treinados[nome] = modelo
        print(f"    {nome} — treino concluído")

    return treinados


# ==============================================================================
# 6. AVALIAÇÃO DOS MODELOS
# ==============================================================================
def avaliar_modelos(modelos, X_test, y_test, feature_names):
    print("\n[6] Avaliando modelos no conjunto de teste...")
    resultados = {}

    for nome, modelo in modelos.items():
        y_pred = modelo.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average="macro")
        f1     = f1_score(y_test, y_pred, average="macro")

        print(f"\n  === {nome} ===")
        print(f"  Acurácia : {acc:.4f}")
        print(f"  Recall (macro) : {recall:.4f}  ← meta ≥ 0.90")
        print(f"  F1-Score (macro): {f1:.4f}")
        print(f"\n  Relatório por classe:")
        print(classification_report(
            y_test, y_pred,
            target_names=[ROTULOS_FALHA[i] for i in sorted(ROTULOS_FALHA)],
        ))
        resultados[nome] = {
            "modelo": modelo, "y_pred": y_pred,
            "acc": acc, "recall": recall, "f1": f1,
        }

    return resultados


# ==============================================================================
# 7. MATRIZES DE CONFUSÃO
# ==============================================================================
def plotar_matrizes_confusao(resultados, y_test):
    print("\n[7] Gerando matrizes de confusão...")
    rotulos = [ROTULOS_FALHA[i] for i in sorted(ROTULOS_FALHA)]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, (nome, res) in zip(axes, resultados.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rotulos)
        disp.plot(ax=ax, colorbar=False, cmap="Blues",
                  values_format="d", xticks_rotation=30)
        ax.set_title(f"{nome}\nAcc={res['acc']:.3f} | Recall={res['recall']:.3f}")

    plt.suptitle("Matrizes de Confusão — Classificação de Falhas", fontsize=14,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    caminho = os.path.join(OUT_DIR, "ML1_matrizes_confusao.png")
    fig.savefig(caminho, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [OK] ML1_matrizes_confusao.png")


# ==============================================================================
# 8. COMPARATIVO DE MÉTRICAS ENTRE MODELOS
# ==============================================================================
def plotar_comparativo_metricas(resultados):
    print("\n[8] Gerando comparativo de métricas...")
    nomes    = list(resultados.keys())
    accs     = [resultados[n]["acc"]    for n in nomes]
    recalls  = [resultados[n]["recall"] for n in nomes]
    f1s      = [resultados[n]["f1"]     for n in nomes]

    x     = np.arange(len(nomes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, accs,    width, label="Acurácia",       color="#1abc9c", edgecolor="black", linewidth=0.5)
    ax.bar(x,         recalls, width, label="Recall (macro)",  color="#3498db", edgecolor="black", linewidth=0.5)
    ax.bar(x + width, f1s,     width, label="F1-Score (macro)", color="#9b59b6", edgecolor="black", linewidth=0.5)

    ax.axhline(0.90, color="#e74c3c", linestyle="--", linewidth=1.5, label="Meta Recall ≥ 90%")

    for i, (a, r, f) in enumerate(zip(accs, recalls, f1s)):
        ax.text(i - width, a + 0.003, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(i,         r + 0.003, f"{r:.3f}", ha="center", fontsize=9)
        ax.text(i + width, f + 0.003, f"{f:.3f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(nomes, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Comparativo de Métricas — Random Forest vs. XGBoost vs. LightGBM")
    ax.legend(loc="lower right")
    ax.yaxis.grid(True, alpha=0.4)

    caminho = os.path.join(OUT_DIR, "ML2_comparativo_metricas.png")
    fig.savefig(caminho, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [OK] ML2_comparativo_metricas.png")


# ==============================================================================
# 9. RECALL POR CLASSE (prioridade operacional)
# ==============================================================================
def plotar_recall_por_classe(resultados, y_test):
    print("\n[9] Gerando Recall por classe de falha...")
    classes   = [ROTULOS_FALHA[i] for i in sorted(ROTULOS_FALHA)]
    n_modelos = len(resultados)
    x         = np.arange(len(classes))
    width      = 0.25
    cores_mod  = ["#1abc9c", "#e67e22", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (nome, res) in enumerate(resultados.items()):
        recalls_por_classe = recall_score(
            y_test, res["y_pred"], average=None,
            labels=sorted(ROTULOS_FALHA.keys()))
        offset = (i - n_modelos / 2 + 0.5) * width
        bars = ax.bar(x + offset, recalls_por_classe, width,
                      label=nome, color=cores_mod[i],
                      edgecolor="black", linewidth=0.4, alpha=0.85)
        for bar, val in zip(bars, recalls_por_classe):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", fontsize=8)

    ax.axhline(0.90, color="#e74c3c", linestyle="--", linewidth=1.5,
               label="Meta Recall ≥ 90%")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Recall por Classe")
    ax.set_title("Recall por Classe de Falha — Prioridade Operacional\n"
                 "(Recall de Desgaste e Desalinhamento são críticos)")
    ax.legend(loc="lower right")
    ax.yaxis.grid(True, alpha=0.4)

    caminho = os.path.join(OUT_DIR, "ML3_recall_por_classe.png")
    fig.savefig(caminho, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [OK] ML3_recall_por_classe.png")


# ==============================================================================
# 10. FEATURE IMPORTANCES (melhor modelo)
# ==============================================================================
def plotar_feature_importances(melhor_modelo, nome_modelo, feature_names):
    print(f"\n[10] Gerando Feature Importances ({nome_modelo})...")
    importances = melhor_modelo.feature_importances_
    idx = np.argsort(importances)[::-1][:20]  # top 20

    fig, ax = plt.subplots(figsize=(12, 7))
    cores = ["#e74c3c" if i == idx[0] else "#3498db" for i in idx]
    ax.barh([feature_names[i] for i in idx[::-1]],
            importances[idx[::-1]],
            color=cores[::-1], edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Importância")
    ax.set_title(f"Top 20 Feature Importances — {nome_modelo}")
    ax.grid(True, alpha=0.3, axis="x")

    caminho = os.path.join(OUT_DIR, f"ML4_feature_importances_{nome_modelo.replace(' ','_')}.png")
    fig.savefig(caminho, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [OK] ML4_feature_importances_{nome_modelo.replace(' ','_')}.png")


# ==============================================================================
# 11. SHAP VALUES (melhor modelo — XGBoost ou LightGBM)
# ==============================================================================
def plotar_shap(modelo, X_test_sc, feature_names, nome_modelo):
    print(f"\n[11] Gerando SHAP Values ({nome_modelo})...")
    X_df = pd.DataFrame(X_test_sc, columns=feature_names)

    explainer   = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(X_df)

    # --- SHAP Summary Plot (beeswarm) ---
    fig, ax = plt.subplots(figsize=(12, 8))
    # shap_values pode ser lista de arrays (multiclasse) ou array 3D
    if isinstance(shap_values, list):
        shap_vals_plot = np.abs(np.array(shap_values)).mean(axis=0)
    else:
        shap_vals_plot = np.abs(shap_values).mean(axis=-1) if shap_values.ndim == 3 else shap_values
    shap.summary_plot(shap_vals_plot, X_df, plot_type="bar",
                      show=False, max_display=20, color="#3498db")
    plt.title(f"SHAP — Importância Global das Features ({nome_modelo})",
              fontsize=12, fontweight="bold")
    caminho = os.path.join(OUT_DIR, f"ML5_shap_summary_{nome_modelo.replace(' ','_')}.png")
    plt.savefig(caminho, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  [OK] ML5_shap_summary_{nome_modelo.replace(' ','_')}.png")


# ==============================================================================
# 12. MÓDULO DE CÁLCULO PREDITIVO DE OEE
# ==============================================================================
def calcular_oee_preditivo(df, modelos, X_full_sc, feature_names):
    """
    Estima o impacto do estado PREDITO no OEE:
    - Tipo 1 (Desgaste) e Tipo 3 (Desalinhamento) → penalizam Disponibilidade
      (risco iminente de parada)
    - Tipo 2 (Falha Refrigeração) → penaliza Qualidade
      (divergência de vazão/pressão → peça com defeito)
    """
    print("\n[12] Calculando OEE Preditivo...")

    # Usamos o melhor modelo (LightGBM) para predição em todo o dataset
    melhor = modelos["LightGBM"]
    preds  = melhor.predict(X_full_sc)
    df     = df.copy()
    df["tipo_falha_predito"] = preds

    # ---- Bases do OEE real (Etapa 1 como baseline) ----------------------
    # Disponibilidade (base): % de ciclos sem falha de parada
    df["flag_parada"]    = df["tipo_falha_predito"].isin([1, 3]).astype(int)
    df["flag_qualidade"] = df["tipo_falha_predito"].isin([2]).astype(int)

    df["hora"] = df["timestamp"].dt.floor("h")
    oee_h = df.groupby(["id_celula", "hora"]).agg(
        disponibilidade=("flag_parada",    lambda x: 1 - x.mean()),
        qualidade      =("flag_qualidade", lambda x: 1 - x.mean()),
        performance    =("vazao_refrigerante_lmin",
                         lambda x: (x / 40.0).clip(0, 1).mean()),
    ).reset_index()
    oee_h["OEE_predito"] = (oee_h["disponibilidade"] *
                             oee_h["performance"] *
                             oee_h["qualidade"])

    # ---- Gráfico H1: OEE preditivo por célula ---------------------------
    oee_cel = oee_h.groupby("id_celula").agg(
        OEE_pred  =("OEE_predito",    "mean"),
        Disp      =("disponibilidade","mean"),
        Perf      =("performance",    "mean"),
        Qual      =("qualidade",      "mean"),
    ) * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    cores_cel = ["#1abc9c", "#9b59b6", "#3498db", "#e74c3c"]

    # OEE total
    ax = axes[0]
    bars = ax.bar(oee_cel.index, oee_cel["OEE_pred"], color=cores_cel,
                  edgecolor="black", linewidth=0.5)
    ax.axhline(85, color="#2ecc71", linestyle="--", linewidth=1.5, label="Meta (85%)")
    ax.axhline(60, color="#e74c3c", linestyle="--", linewidth=1.5, label="Média típica (60%)")
    for bar, val in zip(bars, oee_cel["OEE_pred"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_title("OEE Preditivo por Célula\n(baseado nas predições do LightGBM)")
    ax.set_ylabel("OEE (%)")
    ax.set_ylim(0, 108)
    ax.legend(fontsize=9)

    # Componentes agrupados
    ax2 = axes[1]
    x      = np.arange(len(oee_cel))
    width  = 0.25
    ax2.bar(x - width, oee_cel["Disp"], width, label="Disponibilidade",
            color="#1abc9c", edgecolor="black", linewidth=0.4)
    ax2.bar(x,          oee_cel["Perf"], width, label="Performance",
            color="#3498db", edgecolor="black", linewidth=0.4)
    ax2.bar(x + width,  oee_cel["Qual"], width, label="Qualidade",
            color="#9b59b6", edgecolor="black", linewidth=0.4)
    ax2.set_xticks(x)
    ax2.set_xticklabels(oee_cel.index)
    ax2.set_ylabel("Percentual (%)")
    ax2.set_title("Componentes do OEE Preditivo por Célula\n"
                  "Falha 1/3 → penaliza Disp. | Falha 2 → penaliza Qualidade")
    ax2.legend()
    ax2.set_ylim(0, 115)
    for i in range(len(oee_cel)):
        ax2.text(i - width, oee_cel["Disp"].iloc[i] + 0.5,
                 f"{oee_cel['Disp'].iloc[i]:.1f}%", ha="center", fontsize=8)
        ax2.text(i,          oee_cel["Perf"].iloc[i] + 0.5,
                 f"{oee_cel['Perf'].iloc[i]:.1f}%", ha="center", fontsize=8)
        ax2.text(i + width,  oee_cel["Qual"].iloc[i] + 0.5,
                 f"{oee_cel['Qual'].iloc[i]:.1f}%", ha="center", fontsize=8)

    plt.tight_layout()
    caminho = os.path.join(OUT_DIR, "ML6_oee_preditivo.png")
    fig.savefig(caminho, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [OK] ML6_oee_preditivo.png")

    # ---- Gráfico: Série temporal do OEE preditivo -----------------------
    fig, ax = plt.subplots(figsize=(16, 5))
    for i, cel in enumerate(sorted(oee_h["id_celula"].unique())):
        sub = oee_h[oee_h["id_celula"] == cel]
        ax.plot(sub["hora"], sub["OEE_predito"] * 100,
                color=cores_cel[i], linewidth=0.7, alpha=0.7, label=cel)
    ax.axhline(85, color="#2ecc71", linestyle="--", linewidth=1.5, label="Meta (85%)")
    ax.axhline(60, color="#e74c3c", linestyle="--", linewidth=1.5, label="Média típica (60%)")
    ax.set_ylabel("OEE Preditivo (%)")
    ax.set_xlabel("Data/Hora")
    ax.set_title("Série Temporal — OEE Preditivo por Célula (LightGBM)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    caminho2 = os.path.join(OUT_DIR, "ML7_oee_preditivo_temporal.png")
    fig.savefig(caminho2, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [OK] ML7_oee_preditivo_temporal.png")

    return oee_h


# ==============================================================================
# 13. EXPORTAR MODELOS E PIPELINE (joblib)
# ==============================================================================
def exportar_pipeline(modelos, scaler, feature_names, resultados=None, melhor_nome=None):
    print("\n[13] Exportando modelos e pipeline via joblib...")
    pipeline = {
        "scaler"        : scaler,
        "feature_names" : feature_names,
        "modelos"       : {k: v for k, v in modelos.items()},
        "rotulos_falha" : ROTULOS_FALHA,
        "modelo_selecionado" : melhor_nome,
        "metricas_validacao_temporal" : {
            nome: {"accuracy": res["acc"], "recall_macro": res["recall"], "f1_macro": res["f1"]}
            for nome, res in (resultados or {}).items()
        },
        "criterio_selecao" : "Maior recall macro no conjunto de teste temporal.",
    }
    caminho = os.path.join(OUT_DIR, "vulcantech_pipeline.joblib")
    joblib.dump(pipeline, caminho)
    size_kb = os.path.getsize(caminho) // 1024
    print(f"  [OK] vulcantech_pipeline.joblib ({size_kb} KB) salvo em:")
    print(f"       {caminho}")


# ==============================================================================
# FUNÇÃO PRINCIPAL
# ==============================================================================
def main():
    # -------- 1. Dados --------------------------------------------------------
    df = carregar_dados()

    # -------- Features selecionadas para o modelo ----------------------------
    # Sensores brutos + features derivadas mais relevantes
    FEATURES = [
        # Sensores brutos
        "temperatura_fuso_c", "vibracao_triaxial_rms_mms", "torque_fuso_nm",
        "vazao_refrigerante_lmin", "corrente_estator_amp",
        "pressao_hidraulica_bar", "nivel_ruido_db",
        # Rolling features
        "temperatura_fuso_c_media_30min",     "temperatura_fuso_c_desvio_30min",
        "vibracao_triaxial_rms_mms_media_30min","vibracao_triaxial_rms_mms_desvio_30min",
        "torque_fuso_nm_media_30min",          "torque_fuso_nm_desvio_30min",
        "vazao_refrigerante_lmin_media_2h",    "vazao_refrigerante_lmin_desvio_2h",
        "corrente_estator_amp_media_30min",    "corrente_estator_amp_desvio_30min",
        "pressao_hidraulica_bar_media_2h",     "pressao_hidraulica_bar_desvio_2h",
        "nivel_ruido_db_media_2h",             "nivel_ruido_db_desvio_2h",
        # Features derivadas
        "IEM",
        "temperatura_fuso_c_norm",
        "vibracao_triaxial_rms_mms_norm",
        "torque_fuso_nm_norm",
        "corrente_estator_amp_norm",
        # Encodings categóricos
        "id_celula_enc", "liga_metalica_enc",
    ]

    # -------- 2. Divisão temporal -------------------------------------------
    X_train, X_test, y_train, y_test = dividir_temporal(df, FEATURES)

    # -------- 3. SMOTE -------------------------------------------------------
    X_train_sm, y_train_sm = aplicar_smote(X_train, y_train)

    # -------- 4. Normalização ------------------------------------------------
    X_train_sc, X_test_sc, scaler = normalizar(X_train_sm, X_test)

    # Escalar dataset completo para OEE preditivo
    X_full_sc = scaler.transform(df[FEATURES])

    # -------- 5. Treino ------------------------------------------------------
    modelos = treinar_modelos(X_train_sc, y_train_sm)

    # -------- 6. Avaliação ---------------------------------------------------
    resultados = avaliar_modelos(modelos, X_test_sc, y_test, FEATURES)

    # -------- 7–9. Gráficos de avaliação ------------------------------------
    plotar_matrizes_confusao(resultados, y_test)
    plotar_comparativo_metricas(resultados)
    plotar_recall_por_classe(resultados, y_test)

    # -------- 10. Feature Importances (melhor Recall macro) -----------------
    melhor_nome = max(resultados, key=lambda n: resultados[n]["recall"])
    print(f"\n  ★ Melhor modelo por Recall macro: {melhor_nome} "
          f"({resultados[melhor_nome]['recall']:.4f})")
    plotar_feature_importances(
        resultados[melhor_nome]["modelo"], melhor_nome, FEATURES)

    # -------- 11. SHAP (XGBoost — melhor compatível com TreeExplainer) ------
    shap_modelo = "XGBoost"
    plotar_shap(resultados[shap_modelo]["modelo"],
                X_test_sc, FEATURES, shap_modelo)

    # -------- 12. OEE Preditivo ---------------------------------------------
    oee_h = calcular_oee_preditivo(df, modelos, X_full_sc, FEATURES)

    # -------- 13. Exportar pipeline -----------------------------------------
    exportar_pipeline(modelos, scaler, FEATURES, resultados, melhor_nome)

    # -------- Resumo final --------------------------------------------------
    arquivos = [f for f in os.listdir(OUT_DIR) if f.endswith((".png", ".joblib"))]
    print("\n" + "=" * 65)
    print("  PIPELINE CONCLUÍDO")
    print("=" * 65)
    print(f"  {len(arquivos)} arquivos gerados em:")
    print(f"  {OUT_DIR}")
    for f in sorted(arquivos):
        size = os.path.getsize(os.path.join(OUT_DIR, f)) // 1024
        print(f"    {f} ({size} KB)")
    print("=" * 65)


if __name__ == "__main__":
    main()
