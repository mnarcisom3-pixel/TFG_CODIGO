"""
Utilidades compartidas entre compare_gwas_quanti.py y compare_gwas_quali.py.

Contiene las funciones que eran idénticas en ambos scripts (carga de
resultados, cálculo de métricas, gráfica de comparación), más las utilidades
de localización de fichero de PLINK y de tabla formateada para la memoria.
"""

import os

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

# ---------------------------------------------------------------------------
# 1. Funciones para cargar tus propios resultados y los de PLINK2
# ---------------------------------------------------------------------------

def load_own_results(path):
    d = np.load(path)
    return {"beta": d["beta"], "SE": d["SE"], "p_val": d["p_val"]}


def load_plink_glm(path):
    df = pd.read_csv(path, sep="\t")
    # nos quedamos solo con la fila de efecto aditivo del SNP (no covariables)
    df = df[df["TEST"] == "ADD"].copy()
    # recuperamos el índice numérico del SNP a partir de su ID (SNP_00000, SNP_00001...)
    df["snp_idx"] = df["ID"].str.replace("SNP_", "", regex=False).astype(int)
    df = df.sort_values("snp_idx").reset_index(drop=True)

    # ---------------------------------------------------------------------------
    # 2. Comprobar el alelo contado (A1) antes de comparar betas
    # ---------------------------------------------------------------------------
    # En nuestro VCF, ALT ("T") es siempre el alelo cuya dosis (0/1/2) usamos en
    # matriz012. Si PLINK ha elegido A1 == ALT para un SNP, el beta es directamente
    # comparable. Si PLINK ha elegido A1 == REF, el beta de PLINK viene con el
    # signo invertido respecto al tuyo para ese SNP.

    # alinear el signo del beta: si PLINK ha elegido A1 == REF en vez de ALT,
    # su beta viene invertido respecto a nuestra convención (dosis = copias de ALT)
    sign_flip = np.where(df["ALT1"] == df["A1"], 1.0, -1.0) if "A1" in df.columns else 1.0
    df["beta_aligned"] = df["BETA"] * sign_flip
    return df


# Función para calcular las métricas de correlación entre ambos resultados
def compute_metrics(own, plink_df, label, causal_snp_idx):
    beta_o, beta_p = own["beta"], plink_df["beta_aligned"].to_numpy()
    se_o, se_p = own["SE"], plink_df["SE"].to_numpy()
    p_o = np.clip(own["p_val"], 1e-300, 1)
    p_p = np.clip(plink_df["P"].to_numpy(), 1e-300, 1)
    logp_o, logp_p = -np.log10(p_o), -np.log10(p_p)

    top10_own = set(np.argsort(p_o)[:10])
    top10_plink = set(plink_df.sort_values("P").head(10)["snp_idx"])

    return {
        "escenario": label,
        "pearson_r_beta": np.corrcoef(beta_o, beta_p)[0, 1],
        "pearson_r_SE": np.corrcoef(se_o, se_p)[0, 1],
        "pearson_r_-log10(p)": np.corrcoef(logp_o, logp_p)[0, 1],
        "spearman_r_p": stats.spearmanr(p_o, p_p).correlation,
        "RMSE_beta": np.sqrt(np.mean((beta_o - beta_p) ** 2)),
        "MAE_SE": np.mean(np.abs(se_o - se_p)),
        "concordancia_signo_beta_%": np.mean(np.sign(beta_o) == np.sign(beta_p)) * 100,
        "solapamiento_top10_p": len(top10_own & top10_plink),
        "SNPs_causales_en_top10_propio": len(set(causal_snp_idx) & top10_own),
        "SNPs_causales_en_top10_plink": len(set(causal_snp_idx) & top10_plink),
    }


# Función para crear un gráfico de dispersión que compare ambos resultados
def plot_comparison(own, plink_df, title, outfile, causal_snp_idx=None, beta_label="beta"):
    beta_o, beta_p = own["beta"], plink_df["beta_aligned"].to_numpy()
    se_o, se_p = own["SE"], plink_df["SE"].to_numpy()
    logp_o = -np.log10(np.clip(own["p_val"], 1e-300, 1))
    logp_p = -np.log10(np.clip(plink_df["P"].to_numpy(), 1e-300, 1))

    causal_snp_idx = causal_snp_idx or []
    n = len(beta_o)
    is_causal = np.isin(np.arange(n), list(causal_snp_idx))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = [
        (beta_o, beta_p, beta_label),
        (se_o, se_p, "SE"),
        (logp_o, logp_p, "-log10(p)"),
    ]
    for ax, (x, y, name) in zip(axes, panels):
        r = np.corrcoef(x, y)[0, 1]

        # puntos no causales
        ax.scatter(x[~is_causal], y[~is_causal], s=10, alpha=0.35, color="#4c72b0",
                   label="SNPs sin efecto", rasterized=True)
        # puntos causales, resaltados
        if is_causal.any():
            ax.scatter(x[is_causal], y[is_causal], s=45, alpha=0.9, color="#c44e52",
                       edgecolor="black", linewidth=0.5, label="SNPs causales", zorder=3)

        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lims, lims, "k--", linewidth=1, alpha=0.6, label="y = x")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"{name} (propio)")
        ax.set_ylabel(f"{name} (PLINK2)")
        ax.set_title(name)
        ax.text(
            0.05, 0.95, f"r = {r:.4f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#999999", alpha=0.85),
        )
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.savefig(outfile.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Guardada figura: {outfile} (+ versión .pdf)")


# ---------------------------------------------------------------------------
# 3. Localización automática del fichero de salida de PLINK2 (opcional)
# ---------------------------------------------------------------------------

def find_glm_output(prefix, pheno_name="PHENO"):
    """
    Prueba los sufijos posibles que puede generar PLINK2 según el tipo de
    regresión (lineal, logística, o logística con fallback de Firth) y
    devuelve el primero que exista. Útil si no quieres tener que comprobar
    a mano el sufijo exacto cada vez que corres PLINK.
    """
    candidates = [
        f"{prefix}.{pheno_name}.glm.linear",
        f"{prefix}.{pheno_name}.glm.logistic",
        f"{prefix}.{pheno_name}.glm.logistic.hybrid",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"No se encontró ningún fichero de salida de PLINK para el prefijo "
        f"'{prefix}'. Se buscaron: {candidates}"
    )


# ---------------------------------------------------------------------------
# 4. Tabla de métricas formateada para la memoria (opcional)
# ---------------------------------------------------------------------------

# Nombres legibles para la tabla de la memoria (ver explicación de cada
# métrica en el mensaje de chat)
NICE_LABELS = {
    "pearson_r_beta": "r de Pearson (β)",
    "pearson_r_SE": "r de Pearson (SE)",
    "pearson_r_-log10(p)": "r de Pearson (-log10 p)",
    "spearman_r_p": "r de Spearman (p)",
    "RMSE_beta": "RMSE (β)",
    "MAE_SE": "MAE (SE)",
    "concordancia_signo_beta_%": "Concordancia de signo β (%)",
    "solapamiento_top10_p": "Solapamiento top-10 SNPs",
    "SNPs_causales_en_top10_propio": "SNPs causales en top-10 (propio)",
    "SNPs_causales_en_top10_plink": "SNPs causales en top-10 (PLINK)",
}


def render_metrics_table(df, outfile, title=None):
    """
    df: DataFrame indexado por escenario (filas = escenarios, columnas =
    métricas), tal como lo devuelve pd.DataFrame(rows).set_index("escenario").
    Genera una imagen PNG con la tabla transpuesta (métricas como filas,
    escenarios como columnas), con cabecera resaltada y filas alternas.
    """
    df_t = df.rename(columns=NICE_LABELS).T
    df_t.index.name = "Métrica"
    df_t = df_t.reset_index()

    n_rows, n_cols = df_t.shape
    fig, ax = plt.subplots(figsize=(2.0 + 2.2 * (n_cols - 1), 0.55 * n_rows + 1))
    ax.axis("off")

    table = ax.table(
        cellText=df_t.values,
        colLabels=df_t.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.7)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#2c3e50")
        else:
            cell.set_facecolor("#eef2f7" if row % 2 == 0 else "white")
        cell.set_edgecolor("#cccccc")
        if col == 0:
            cell.set_text_props(ha="left", weight="bold")

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=18)

    plt.tight_layout()
    plt.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardada tabla: {outfile}")
