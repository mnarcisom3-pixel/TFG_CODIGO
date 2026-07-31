r"""
Compara tu función linreg_3d con la salida de PLINK2, en dos escenarios:

  1. "mismos_PCs"  : PLINK usa tus PCs de pynei como covariables
                     (fichero: gwas_plink2_ownpcs.PHENO.glm.linear)
  2. "pca_plink"   : PLINK calcula su propio PCA (con --pca) y lo usa como
                     covariable (fichero: gwas_plink2_plinkpca.PHENO.glm.linear)

Genera:
  - comparacion_mismos_PCs.png
  - comparacion_pca_plink.png
  - tabla_metricas_validacion.csv   (para pegar/formatear en la memoria)

--------------------------------------------------------------------------
El mismo gwas_quanti.vcf y gwas_quanti_pheno.txt sirven para los dos
escenarios; solo cambia el fichero de covariables pasado a --glm.

Escenario 1 (mismos PCs, ya generado por export_quanti.py como
gwas_quanti_covar.txt) — es el comando que ya corriste, renombrando el --out
a gwas_plink2_ownpcs para distinguirlo del segundo:

.\plink2.exe --vcf gwas_quanti.vcf --double-id `
    --pheno gwas_quanti_pheno.txt --pheno-name PHENO `
    --covar gwas_quanti_own_covars.txt `
    --glm hide-covar cols=chrom,pos,ref,alt1,test,nobs,beta,se,p `
    --out gwas_plink2_quanti_ownpcs

Escenario 2 (PCA calculado por PLINK), sobre el mismo VCF:

    # 1. Calcular PCA con PLINK (10 componentes)
.\plink2.exe --vcf gwas_quanti.vcf --double-id --pca 10 --out gwas_quanti_pca_from_plink

    # 2. GWAS usando los PCs de PLINK como covariables
.\plink2.exe --vcf gwas_quanti.vcf --double-id `
    --pheno gwas_quanti_pheno.txt --pheno-name PHENO `
    --covar gwas_quanti_pca_from_plink.eigenvec `
    --glm hide-covar cols=chrom,pos,ref,alt1,test,nobs,beta,se,p `
    --out gwas_plink2_quanti_plinkpca
--------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

OWN_RESULTS_PATH = "own_results_quanti.npz"
PLINK_FILES = {
    "mismos_PCs": "gwas_plink2_quanti_ownpcs.PHENO.glm.linear",
    "pca_plink": "gwas_plink2_quanti_plinkpca.PHENO.glm.linear",
}
CAUSAL_SNP_IDX = [0, 1, 2, 3, 4]  # los SNPs que introdujiste con efecto real

# ---------------------------------------------------------------------------
# 1. Funciones para cargar tus propios resultados (linreg_3d) y los de PLINK2
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
def compute_metrics(own, plink_df, label):
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
        "SNPs_causales_en_top10_propio": len(set(CAUSAL_SNP_IDX) & top10_own),
        "SNPs_causales_en_top10_plink": len(set(CAUSAL_SNP_IDX) & top10_plink),
    }

# Función para crear un gráfico de dispersión que compare ambos resultados
def plot_comparison(own, plink_df, title, outfile):
    beta_o, beta_p = own["beta"], plink_df["beta_aligned"].to_numpy()
    se_o, se_p = own["SE"], plink_df["SE"].to_numpy()
    logp_o = -np.log10(np.clip(own["p_val"], 1e-300, 1))
    logp_p = -np.log10(np.clip(plink_df["P"].to_numpy(), 1e-300, 1))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = [
        (beta_o, beta_p, "beta"),
        (se_o, se_p, "SE"),
        (logp_o, logp_p, "-log10(p)"),
    ]
    for ax, (x, y, name) in zip(axes, panels):
        ax.scatter(x, y, s=8, alpha=0.5)
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_xlabel(f"{name} (propio)")
        ax.set_ylabel(f"{name} (PLINK2)")
        ax.set_title(name)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"Guardada figura: {outfile}")


if __name__ == "__main__":
    own_results = load_own_results(OWN_RESULTS_PATH)

    rows = []
    for label, path in PLINK_FILES.items():
        plink_df = load_plink_glm(path)
        rows.append(compute_metrics(own_results, plink_df, label))
        plot_comparison(own_results, plink_df, f"Validación GWAS cuantitativo — {label}", f"comparacion_QUANTI_{label}.png")

    tabla = pd.DataFrame(rows).set_index("escenario").round(4)
    tabla.to_csv("tabla_metricas_validacion_QUANTI.csv")
    print("\n", tabla.to_string())
