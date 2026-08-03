r"""
Compara tu función cualitativa con la salida de PLINK2 (regresión logística),
en dos escenarios: mismos PCs vs PCA calculado por PLINK. Ver el docstring de
compare_gwas_quanti.py para más detalle del proceso; aquí solo se apuntan las
diferencias propias del caso binario.

--------------------------------------------------------------------------
El mismo gwas_quali.vcf y gwas_quali_pheno.txt sirven para los dos escenarios.
Nota el --1 (fenotipo 0=control/1=caso) y que el fichero de salida tiene
sufijo .glm.logistic (o .glm.logistic.hybrid si algún SNP recurrió a
regresión de Firth; comprueba con `dir` el nombre exacto):

    # Escenario 1: con tus propios PCs (ya generados por export_quali.py
    # como gwas_quali_covar.txt)
.\plink2.exe --vcf gwas_quali.vcf --double-id --1 `
    --pheno gwas_quali_pheno.txt --pheno-name PHENO `
    --covar gwas_quali_own_covars.txt `
    --glm hide-covar cols=chrom,pos,ref,alt1,test,nobs,beta,se,p `
    --out gwas_plink2_quali_ownpcs

    # Escenario 2: PCA calculado por PLINK, sobre el mismo VCF
.\plink2.exe --vcf gwas_quali.vcf --double-id --pca 10 --out gwas_quali_pca_from_plink

.\plink2.exe --vcf gwas_quali.vcf --double-id --1 `
    --pheno gwas_quali_pheno.txt --pheno-name PHENO `
    --covar gwas_quali_pca_from_plink.eigenvec `
    --glm hide-covar cols=chrom,pos,ref,alt1,test,nobs,beta,se,p `
    --out gwas_plink2_quali_plinkpca
--------------------------------------------------------------------------
"""
import pandas as pd
 
from validation_utils import (
    load_own_results,
    load_plink_glm,
    compute_metrics,
    plot_comparison,
    render_metrics_table,
    find_glm_output
)
 
OWN_RESULTS_PATH = "own_results_quali.npz"
PLINK_FILES = {
    "mismos_PCs": "gwas_plink2_quali_ownpcs.PHENO.glm.logistic.hybrid",
    "pca_plink": "gwas_plink2_quali_plinkpca.PHENO.glm.logistic.hybrid",
}
CAUSAL_SNP_IDX = [0, 1, 2, 3, 4]
 
 
if __name__ == "__main__":
    own_results = load_own_results(OWN_RESULTS_PATH)
 
    rows = []
    for label, path in PLINK_FILES.items():
        plink_df = load_plink_glm(path)
        rows.append(compute_metrics(own_results, plink_df, label, CAUSAL_SNP_IDX))
        plot_comparison(
            own_results, plink_df,
            f"Validación GWAS cualitativo — {label}",
            f"comparacion_QUALI_{label}.png",
            causal_snp_idx=CAUSAL_SNP_IDX,
            beta_label="beta (log-odds)",
        )
 
    tabla = pd.DataFrame(rows).set_index("escenario").round(4)
    tabla.to_csv("tabla_metricas_validacion_QUALI.csv")
    render_metrics_table(tabla, "tabla_metricas_validacion_QUALI.png", title="Validación GWAS cualitativo vs. PLINK2")
    print("\n", tabla.to_string())