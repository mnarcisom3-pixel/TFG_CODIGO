"""
Genera los datos de juguete cuantitativos y los exporta a formato PLINK2.
Necesita plink_io.py en el mismo directorio (o en el PYTHONPATH).
"""
from pathlib import Path

import numpy as np
import pandas as pd

import gwaslib.quantitative as gw_quant
from gwaslib import integration
import pynei

from plink_io import export_to_plink, export_to_plink_with_real_positions

# CON DATOS DE JUGUETE
'''
# ---------------------------------------------------------------------------
# 1. Generación de datos (idéntico a tu script original)
# ---------------------------------------------------------------------------
np.random.seed(12345)

M_snps = 10000
N_indivs = 200
K_pcs = 10

p_alt = np.random.uniform(0.05, 0.5, size=(M_snps, 1))
prob_0 = (1 - p_alt) ** 2
prob_1 = 2 * p_alt * (1 - p_alt)
prob_2 = p_alt**2
probs = np.hstack([prob_0, prob_1, prob_2])

matriz012_test = np.empty((M_snps, N_indivs))
for idx_snp in range(M_snps):
    matriz012_test[idx_snp, :] = np.random.choice(
        [0, 1, 2], size=N_indivs, p=probs[idx_snp]
    )

mat012_forPCA = pd.DataFrame(matriz012_test.T)
PCA = pynei.pca.do_pca(mat012_forPCA)
covariables_test = PCA["projections"].to_numpy()[:, :10]

phenotypes_test = (
    170.0
    + (2.0 * matriz012_test[0, :])
    - (1.5 * matriz012_test[1, :])
    + (2.5 * matriz012_test[2, :])
    - (4.0 * matriz012_test[3, :])
    + (0.7 * matriz012_test[4, :])
    + np.random.normal(loc=0.0, scale=4.0, size=N_indivs)
)
phenotypes_test = np.clip(phenotypes_test, 140.0, 210.0)

# ---------------------------------------------------------------------------
# 2. Exportar a formato PLINK
# ---------------------------------------------------------------------------
# Un único VCF y un único fichero de fenotipo sirven para los dos escenarios
# de validación (mismos PCs / PCA propio de PLINK): lo único que cambia entre
# ambos es qué fichero de covariables se pasa a "--covar" en el --glm. Para
# el escenario "PCA propio de PLINK", el --pca de PLINK se ejecuta sobre este
# mismo gwas_quanti.vcf (no hace falta una versión sin covariables).
sample_ids = [f"IND{i + 1}" for i in range(N_indivs)]
export_to_plink(matriz012_test, phenotypes_test, covariables_test, sample_ids, "gwas_quanti")

# ---------------------------------------------------------------------------
# 3. Guardar mis propios resultados para comparar después
# ---------------------------------------------------------------------------

own_results = gw_quant.linreg_3d(matriz012_test, phenotypes_test, covariables_test)
np.savez("own_results_quanti.npz", **own_results)

'''
# Con datos reales
PROJECT_DIR = Path(__file__).parent.parent.parent

# SELECCIONAR VCF ------------------------------------------------------------------------------------------
# Una versión aún más reducida (20MB) para que el tiempo de computación sea menor
#path_VCF = PROJECT_DIR / "geno_pheno_files" / "VCF_FILES_from_Ximo" / "Varitome_20mb_reduced.vcf"

# El de probar la web, de Varitome reducido a 100MB
path_VCF = PROJECT_DIR / "geno_pheno_files" / "VCF_FILES_from_Ximo" / "Varitome_reduced_all_chroms.vcf"

# LEER VCF CON PYNEI ----------------------------------------------------------------------------------------
variants = pynei.io_vcf.vars_from_vcf(path_VCF)

matriz012_crude = pynei.pca.create_012_gt_matrix(variants, transform_to_biallelic=True)
print(matriz012_crude.shape)

# LEER FICHERO DE FENOTIPOS ---------------------------------------------------------------------------------
# Cuantitativos
path_csv_quanti = PROJECT_DIR / "geno_pheno_files" / "PHENOTYPES_from_Ximo" / "CSV_files" / "quanti_trait_mean_color_b.csv"

df_crude_feno_quanti = integration.load_phenotypes(path_csv_quanti)

# FILTRAR GENOTIPOS Y FENOTIPOS PARA EL GWAS
filtered_vars_for_GWAS = integration.filter_genotypes_for_GWAS(variants, df_crude_feno_quanti)

df_filtered_feno = integration.filter_phenotypes(df_crude_feno_quanti, filtered_vars_for_GWAS.samples)
print(df_filtered_feno.to_numpy().shape)

# Importamos PCA en CSV
PCA_DIR = PROJECT_DIR / "geno_pheno_files" / "PCA_FILES_from_VCFs"
path_PCA = PCA_DIR / "From_100mb_VCF" / "PCA_quanti_trait_mean_color_b.csv"

df_all_pcs = pd.read_csv(path_PCA, index_col=0)

# EJECUTAR GWAS ----------------------------------------------------------------------------------------------
gwas_results = integration.do_gwas(filtered_vars=filtered_vars_for_GWAS,
                filtered_phenotypes=df_filtered_feno,
                covariates=df_all_pcs,
                type_of_phenotype="cuantitativo",      # "cualitativo (binario)",
                sort_by_significance=False,
                )

filtered_vars_for_012 = integration.filter_genotypes_for_GWAS(variants, df_crude_feno_quanti)

real_samples_ids = filtered_vars_for_012.samples
mat012_filtrada = pynei.pca.create_012_gt_matrix(filtered_vars_for_012, transform_to_biallelic=True)
feno_array = df_filtered_feno.to_numpy()
pcs_array = df_all_pcs.to_numpy()[:, :10]

#sample_ids = [f"IND{i + 1}" for i in range(N_indivs)]
#export_to_plink(mat012_filtrada, feno_array, pcs_array, real_samples_ids, "Varitome_filt_20mb_fruit_weight")

export_to_plink_with_real_positions(matriz012=mat012_filtrada, phenotypes=feno_array, covariables=pcs_array, gwas_results=gwas_results, sample_ids=real_samples_ids, out_prefix="Varitome_filt_100mb_mean_color_b")

# Export my own results as csv
gwas_results.to_csv("Varitome_own_results_quanti.csv", index=False)