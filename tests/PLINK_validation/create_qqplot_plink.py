from gwaslib import visualization

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# Parámetros
# ============================================================

# CUANTITATIVO
'''
plink_results_path = (
    "Varitome_gwas_plink2_quanti2_ownpcs.PHENO.glm.linear"
)

gwaslib_results_path = (
    "Varitome_own_results_quanti_2.csv"
)

phenotype_name_plink = "PLINK2 \n  (Mean Color b)"
phenotype_name_gwaslib = "GWASLIB \n  (Mean Color b)"

plink_output_png = "QQ_PLINK_mean_color_b.png"
plink_output_pdf = "QQ_PLINK_mean_color_b.pdf"

gwaslib_output_png = "QQ_GWASLIB_mean_color_b.png"
gwaslib_output_pdf = "QQ_GWASLIB_mean_color_b.pdf"
'''

# CUALITATIVO
plink_results_path = (
    "Varitome_gwas_plink2_quali_ownpcs.PHENO.glm.logistic.hybrid"
)

gwaslib_results_path = (
    "Varitome_own_results_quali.csv"
)

phenotype_name_plink = "PLINK2 \n  (Inflorecence Forked Type)"
phenotype_name_gwaslib = "GWASLIB \n  (Inflorecence Forked Type)"

plink_output_png = "QQ_PLINK_inflorescence_forked.png"
plink_output_pdf = "QQ_PLINK_inflorescence_forked.pdf"

gwaslib_output_png = "QQ_GWASLIB_inflorescence_forked.png"
gwaslib_output_pdf = "QQ_GWASLIB_inflorescence_forked.pdf"


# ============================================================
# 1. Leer resultados PLINK
# ============================================================

df = pd.read_csv(
    plink_results_path,
    sep="\t",
)

'''
invalid = ~np.isfinite(
    pd.to_numeric(df["P"], errors="coerce")
)

print(
    df.loc[invalid, "ERRCODE"]
    .value_counts(dropna=False)
)

# Al correr esto con "plink_results_path = ("ERRCODE_quanti.PHENO.glm.linear")"
# Esto nos devolvió "VIF_TOO_HIGH    2338"
'''

# Solo test aditivo
df = df[
    df["TEST"] == "ADD"
].copy()

'''
p = pd.to_numeric(df["P"], errors="coerce")

print("NaN:", p.isna().sum())
print("P = 0:", (p == 0).sum())
print("P < 0:", (p < 0).sum())
print("P > 1:", (p > 1).sum())
'''

# Seleccionamos solo los SNPs con p-valores válidos (finitos) de PLINK
# Vamos a eliminar los SNPs que tenían p-valores inválidos en PLINK,
# (eliminarlos del dataframe que devuelve do_gwas)
my_gwas_results = pd.read_csv(gwaslib_results_path)

plink_p = pd.to_numeric(
    df["P"],
    errors="coerce",
).to_numpy()

valid_common = (
    np.isfinite(plink_p)
    & (plink_p > 0)
    & (plink_p <= 1)
)

plink_common_snps = df.loc[valid_common].copy()
gwaslib_common_snps = my_gwas_results.loc[valid_common].copy()



print(len(gwaslib_common_snps), len(plink_common_snps))

'''
print("Mediana common gwaslib:", np.median(p_gwaslib_common))
print("Mediana common PLINK:", np.median(p_plink_common))
'''

# ============================================================
# 2. P-valores observados en PLINK
# ============================================================

p_values = pd.to_numeric(
    plink_common_snps["P"],
    errors="coerce",
).to_numpy()

observed_logp = -np.log10(p_values)

observed_logp = np.sort(observed_logp)

num_tests = len(observed_logp)

# ============================================================
# 3. Distribución esperada bajo H0
# ============================================================

expected_p = (
    np.arange(
        1,
        num_tests + 1,
    )
    / (num_tests + 1)
)

expected_logp = -np.log10(
    expected_p
)

expected_logp = np.sort(
    expected_logp
)


# ============================================================
# 4. Lambda GC
# ============================================================

chi2_values = stats.chi2.isf(
    p_values,
    df=1,
)

lambda_gc = (
    np.median(chi2_values)
    / stats.chi2.ppf(
        0.5,
        df=1,
    )
)


# ============================================================
# 5. Q-Q plot PLINK
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 7)
)

ax.scatter(
    expected_logp,
    observed_logp,
    s=10,
    alpha=0.6,
    rasterized=True,
)

max_expected = (
    expected_logp.max()
)

ax.plot(
    [0, max_expected],
    [0, max_expected],
    "k--",
    linewidth=1,
    label="Expected under H₀",
)

ax.set_xlabel(
    "Expected -log10(p)"
)

ax.set_ylabel(
    "Observed -log10(p)"
)

ax.set_title(
    f"Q-Q plot — {phenotype_name_plink}"
)

ax.text(
    0.05,
    0.90,
    rf"$\lambda_{{GC}}$ = {lambda_gc:.3f}",
    transform=ax.transAxes,
    va="top",
)

ax.legend()

ax.set_xlim(left=0)
ax.set_ylim(bottom=0)

fig.tight_layout()

fig.savefig(
    plink_output_png,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    plink_output_pdf,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 6. Q-Q plot gwaslib
# ============================================================

fig, ax = visualization.create_qq_plot(
    gwaslib_common_snps,
    phenotype_name=phenotype_name_gwaslib,
)

fig.savefig(
    gwaslib_output_png,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    gwaslib_output_pdf,
    bbox_inches="tight",
)

plt.close(fig)

print("Q-Q plots generados correctamente.")

