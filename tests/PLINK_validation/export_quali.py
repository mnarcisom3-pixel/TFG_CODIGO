"""
Genera los datos de juguete cualitativos (fenotipo binario) y los exporta a
formato PLINK2. Necesita plink_io.py en el mismo directorio.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from gwaslib import integration
import gwaslib.qualitative as gw_qual
import pynei

from plink_io import export_to_plink, export_to_plink_with_real_positions

'''
# ---------------------------------------------------------------------------
# 1. Generación de datos
# ---------------------------------------------------------------------------
np.random.seed(12345)

M_snps = 10000
N_indivs = 300
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

eta_real = (
    -2.8
    + (1.6 * matriz012_test[0, :])
    - (1.4 * matriz012_test[1, :])
    + (1.5 * matriz012_test[2, :])
    - (1.5 * matriz012_test[3, :])
    + (1.3 * matriz012_test[4, :])
)
prob_real = 1.0 / (1.0 + np.exp(-eta_real))
phenotypes_test = np.random.binomial(n=1, p=prob_real)

# ---------------------------------------------------------------------------
# 2. Exportar a formato PLINK
# ---------------------------------------------------------------------------
# Un único VCF y un único fichero de fenotipo sirven para los dos escenarios
# de validación; el --pca de PLINK se calcula sobre este mismo gwas_quali.vcf.
sample_ids = [f"IND{i + 1}" for i in range(N_indivs)]
export_to_plink(matriz012_test, phenotypes_test, covariables_test, sample_ids, "gwas_quali")

# ---------------------------------------------------------------------------
# 3. Guardar mis propios resultados para comparar después
# ---------------------------------------------------------------------------
own_results = gw_qual.logreg_3d(matriz012_test, phenotypes_test, covariables_test)
np.savez("own_results_quali.npz", **own_results)
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
path_csv_quali = PROJECT_DIR / "geno_pheno_files" / "PHENOTYPES_from_Ximo" / "CSV_files" / "quali_trait_inflorescence_forked_type_clean.csv"

df_crude_feno_quali = integration.load_phenotypes(path_csv_quali)

# FILTRAR GENOTIPOS Y FENOTIPOS PARA EL GWAS
filtered_vars_for_GWAS = integration.filter_genotypes_for_GWAS(variants, df_crude_feno_quali)

df_filtered_feno = integration.filter_phenotypes(df_crude_feno_quali, filtered_vars_for_GWAS.samples)
print(df_filtered_feno.to_numpy().shape)

# Importamos PCA en CSV
PCA_DIR = PROJECT_DIR / "geno_pheno_files" / "PCA_FILES_from_VCFs"
path_PCA = PCA_DIR / "From_100mb_VCF" / "PCA_quali_trait_inflorescence_forked_type_clean.csv"

df_all_pcs = pd.read_csv(path_PCA, index_col=0)

# EJECUTAR GWAS ----------------------------------------------------------------------------------------------
gwas_results = integration.do_gwas(filtered_vars=filtered_vars_for_GWAS,
                filtered_phenotypes=df_filtered_feno,
                covariates=df_all_pcs,
                type_of_phenotype="cualitativo (binario)",
                sort_by_significance=False,
                )

filtered_vars_for_012 = integration.filter_genotypes_for_GWAS(variants, df_crude_feno_quali)

real_samples_ids = filtered_vars_for_012.samples
mat012_filtrada = pynei.pca.create_012_gt_matrix(filtered_vars_for_012, transform_to_biallelic=True)
feno_array = df_filtered_feno.to_numpy()
pcs_array = df_all_pcs.to_numpy()[:, :10]

#sample_ids = [f"IND{i + 1}" for i in range(N_indivs)]
#export_to_plink(mat012_filtrada, feno_array, pcs_array, real_samples_ids, "Varitome_filt_100mb_inflorescence_forked")

export_to_plink_with_real_positions(matriz012=mat012_filtrada, phenotypes=feno_array, covariables=pcs_array, gwas_results=gwas_results, sample_ids=real_samples_ids, out_prefix="Varitome_filt_100mb_inflorescence_forked")

# Export my own results as csv
gwas_results.to_csv("Varitome_own_results_quali.csv", index=False)

'''
> .plink2.exe `                        
>>     --vcf Varitome_filt_20mb_longitudinal_stripes.vcf `                                         
>>     --double-id --1 `
>>     --pheno Varitome_filt_20mb_longitudinal_stripes_pheno.txt --pheno-name PHENO `
>>     --covar Varitome_filt_20mb_longitudinal_stripes_own_covars.txt `
>>     --glm hide-covar firth-fallback cols=chrom,pos,ref,alt1,firth,test,nobs,beta,se,p,err `
>>     --out Varitome_gwas_plink2_quali_ownpcs
'''

