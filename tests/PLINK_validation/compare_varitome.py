import pandas as pd
import numpy as np

from validation_utils import (
    load_plink_glm,
    plot_pvalue_comparison,
    plot_pvalue_comparison_showing_firth
)


# Resultados de do_gwas SIN ordenar
gwas_results = pd.read_csv("Varitome_own_results_quali.csv")

own_results = {
    "beta": gwas_results["beta"].to_numpy(),
    "SE": gwas_results["SE"].to_numpy(),
    "p_val": 10 ** (-gwas_results["-log10(p)"].to_numpy()),
}


# Resultados PLINK
plink_df = load_plink_glm(
    "Varitome_gwas_plink2_quali_ownpcs.PHENO.glm.logistic.hybrid"  # .linear
)


# Seguridad: mismo número y mismo orden
assert len(own_results["p_val"]) == len(plink_df)

print(plink_df[["ID", "P"]].head(10))
print(gwas_results[["Chromosome", "Position", "-log10(p)"]].head(10))

print("PLINK p=0:", np.sum(plink_df["P"].to_numpy() == 0))
print("Propios p=0:", np.sum(own_results["p_val"] == 0))
print("PLINK p=NA:", plink_df["P"].isna().sum())

# Para el CUANTITATIVO
'''
plot_pvalue_comparison(
    own_results,
    plink_df,
    title="PLINK2 vs gwaslib para un dataset cuantitativo real\n(Mean Color b)",
    outfile="Varitome_comparacion_QUANTI_mean_color_b.png",
)
'''

# Para el CUALITATIVO
plot_pvalue_comparison_showing_firth(
    own_results,
    plink_df,
    title="PLINK2 vs gwaslib para un dataset cualitativo real\n(Inflorescence Forked Type)",
    outfile="Varitome_comparacion_QUALI_inflorescence_forked_FIRTH_both_R.png",
)
