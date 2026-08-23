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

# O con los datos de Varitome
.\plink2.exe --vcf Varitome_filt_100mb_mean_color_b.vcf --double-id `
    --pheno Varitome_filt_100mb_mean_color_b_pheno.txt --pheno-name PHENO `
    --covar Varitome_filt_100mb_mean_color_b_own_covars.txt `
    --glm hide-covar cols=chrom,pos,ref,alt1,test,nobs,beta,se,p `
    --out Varitome_gwas_plink2_quanti_ownpcs


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
import pandas as pd

from validation_utils import (
    load_own_results,
    load_plink_glm,
    compute_metrics,
    plot_comparison,
    render_metrics_table,
)
 
OWN_RESULTS_PATH = "own_results_quanti.npz"
PLINK_FILES = {
    "mismos_PCs": "gwas_plink2_quanti_ownpcs.PHENO.glm.linear",
    "pca_plink": "gwas_plink2_quanti_plinkpca.PHENO.glm.linear",
}
CAUSAL_SNP_IDX = [0, 1, 2, 3, 4]  # los SNPs que introdujiste con efecto real
 
 
if __name__ == "__main__":
    own_results = load_own_results(OWN_RESULTS_PATH)
 
    rows = []
    for label, path in PLINK_FILES.items():
        plink_df = load_plink_glm(path)
        rows.append(compute_metrics(own_results, plink_df, label, CAUSAL_SNP_IDX))
        plot_comparison(
            own_results, plink_df,
            f"Validación GWAS cuantitativo — {label}",
            f"comparacion_QUANTI_{label}.png",
            causal_snp_idx=CAUSAL_SNP_IDX,
        )
 
    tabla = pd.DataFrame(rows).set_index("escenario").round(4)
    tabla.to_csv("tabla_metricas_validacion_QUANTI.csv")
    render_metrics_table(tabla, "tabla_metricas_validacion_QUANTI.png", title="Validación GWAS cuantitativo vs. PLINK2")
    print("\n", tabla.to_string())
