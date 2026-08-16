import numpy as np

from pathlib import Path

from gwaslib import integration
import pynei

PROJECT_DIR = Path(__file__).parent.parent.parent

# Los VCF para mis pruebas originales muy pequeños (muy pocos SNPs)
# path_VCF = PROJECT_DIR / "geno_pheno_files" / "vcf_test_247rows.vcf"
# path_VCF = PROJECT_DIR / "geno_pheno_files" / "vcf_7_samples.vcf"

# El de probar la web, de Varitome reducido a 100MB
#path_VCF = PROJECT_DIR / "geno_pheno_files" / "VCF_FILES_from_Ximo" / "Varitome_reduced_all_chroms.vcf"

# Una versión aún más reducida (20MB) para que el tiempo de computación sea menor
path_VCF = PROJECT_DIR / "geno_pheno_files" / "VCF_FILES_from_Ximo" / "Varitome_20mb_reduced.vcf"

# Una versión intermedia (50MB)
#path_VCF = PROJECT_DIR / "geno_pheno_files" / "VCF_FILES_from_Ximo" / "Varitome_50mb_reduced.vcf"


# Leemos el VCF con pynei y sacamos la mat012 sin filtrar nada
variants = pynei.io_vcf.vars_from_vcf(path_VCF)

matriz012_crude = pynei.pca.create_012_gt_matrix(variants, transform_to_biallelic=True)
print(matriz012_crude.shape)
# Para el vcf de 20MB   --> (4122, 143)
# Para el vcf de 50MB   --> (69707, 166)
# Para el vcf de 100MB  --> (27807, 166)

# APRENDIENDO SOBRE EL FUNCIONAMIENTO DE PYNEI
'''
# DUDA 1 = Vamos a comprobar si un objeto Variants (tal cual del VCF) es un iterador de un solo uso
matriz012_crude_2 = pynei.pca.create_012_gt_matrix(variants, transform_to_biallelic=True)
print(matriz012_crude_2.shape)

# Ahora comprobémoslo para un objeto Variants FILTRADO
filtered_variants = pynei.var_filters.filter_by_ld_and_maf(variants)
matriz012_filtrada = pynei.pca.create_012_gt_matrix(filtered_variants, transform_to_biallelic=True)
print(matriz012_filtrada.shape)
matriz012_filtrada_2 = pynei.pca.create_012_gt_matrix(filtered_variants, transform_to_biallelic=True)
print(matriz012_filtrada_2.shape)
'''
'''
# DUDA 2 = Se ha visto que los Filtered_Variants son un iterador de un solo uso
# Si generamos solo uno, aunque le pongamos varios punteros, solo tendrá un uso
filt = pynei.var_filters.filter_by_ld_and_maf(variants)
a = filt
b = filt

mat_a = pynei.pca.create_012_gt_matrix(a, transform_to_biallelic=True)
print(mat_a.shape)
mat_b = pynei.pca.create_012_gt_matrix(b, transform_to_biallelic=True)
print(mat_b.shape)
'''
'''
# ¿Si generamos varios idénticos, llamando a la función de filtrado varias veces sobre el mismo input,
# podemos usarlos todos?

filt_a = pynei.var_filters.filter_by_ld_and_maf(variants)
filt_b = pynei.var_filters.filter_by_ld_and_maf(variants)

mat_a = pynei.pca.create_012_gt_matrix(filt_a, transform_to_biallelic=True)
print(mat_a.shape)
mat_b = pynei.pca.create_012_gt_matrix(filt_b, transform_to_biallelic=True)
print(mat_b.shape)

# Efectivamente, sí que podemos
'''

''' POR TIEMPO
# DUDA 3 = ¿¿Obtener las samples_idx de un Variants filtrado consumirá el iterador??
# Probemos
filt = pynei.var_filters.filter_by_ld_and_maf(variants)
a = filt.samples
print(a)
b = filt.samples
print(b)
# No lo consume
'''


# LO DE ESTE COMENTARIO ES DE LA PRIMERA VEZ QUE USÉ PYNEI (ignorarlo)
'''
# Hagamos el filtrado para el PCA (filtrar por maf y LD)
filtered_variants = pynei.var_filters.filter_by_ld_and_maf(variants)

# Hay alguna manera de mostrar el número de variantes que quedan tras el filtrado??
# Pues mira...
# Probemos a ver el tamaño de la matriz012 que le damos a do_pca
matriz012_sin = pynei.pca.create_012_gt_matrix(variants, transform_to_biallelic=True)
matriz012_filt = pynei.pca.create_012_gt_matrix(filtered_variants, transform_to_biallelic=True)

print(matriz012_sin.shape)
print(matriz012_filt.shape)

# Hagamos el PCA
PCs= pynei.pca.do_pca_with_vars(filtered_variants, transform_to_biallelic=True)
# PCs_sin = pynei.pca.do_pca_with_vars(variants, transform_to_biallelic=True)
#print(PCs)

covariables = PCs["projections"].to_numpy()
comb_lineales = PCs["princomps"].to_numpy()
print(covariables.shape)
print(comb_lineales.shape)

# OJO = La función do_pca recibe una matriz012 TRANSPUESTA (N x SNPs) !!!
# Probemos a correr lo mismo con 500 variantes en lugar de 247
'''
#---------------------------------------------------------------------------------------------------
# Probemos las funciones de integration.py
#---------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------------------------------
# PROBEMOS LA CARGA DE FENOTIPOS
excel_path_feno_quanti = PROJECT_DIR / "geno_pheno_files" / "PHENOTYPES_from_Ximo" / "Excel_files" / "quanti_trait_mean_fruit_weight.xlsx"
csv_path_feno_quanti = PROJECT_DIR / "geno_pheno_files" / "PHENOTYPES_from_Ximo" / "CSV_files" / "quanti_trait_mean_fruit_weight.csv"

crude_feno_quanti_1 = integration.load_phenotypes(excel_path_feno_quanti)
crude_feno_quanti_2 = integration.load_phenotypes(csv_path_feno_quanti)
feno_1 = crude_feno_quanti_1.to_numpy()
feno_2 = crude_feno_quanti_2.to_numpy()
print(feno_1.shape)
print(feno_2.shape)
print(crude_feno_quanti_1.keys().to_numpy().shape) # Los Sample IDs
# Ambos (.xlsx y .csv) funcionan

'''
# Pruebas varias con numpy
feno_array = feno_1.reshape(1, len(feno_1))
print(feno_array.shape)
for i in feno_array:
    print(i)
feno_duplicated = np.vstack((feno_1, feno_2))
print(feno_duplicated.shape)
'''

# Ahora los cualitativos
excel_path_feno_quali = PROJECT_DIR / "geno_pheno_files" / "PHENOTYPES_from_Ximo" / "Excel_files" / "quali_trait_fruit_longitudinal_stripes_clean.xlsx"
csv_path_feno_quali = PROJECT_DIR / "geno_pheno_files" / "PHENOTYPES_from_Ximo" / "CSV_files" / "quali_trait_fruit_longitudinal_stripes_clean.csv"

crude_feno_quali_1 = integration.load_phenotypes(excel_path_feno_quali)
crude_feno_quali_2 = integration.load_phenotypes(csv_path_feno_quali)
feno_3 = crude_feno_quali_1.to_numpy()
feno_4 = crude_feno_quali_2.to_numpy()
print(feno_3.shape)
print(feno_4.shape)
print(crude_feno_quali_1.keys().to_numpy().shape)  # Los Sample IDs
# Ambos (.xlsx y .csv) funcionan

# Probemos la identificación de fenotipos "no NaN" (fenotipos conocidos)
samples_known_pheno_quanti = integration._get_samples_with_known_phenotypes(crude_feno_quanti_1)
print(samples_known_pheno_quanti.shape)

samples_known_pheno_quali = integration._get_samples_with_known_phenotypes(crude_feno_quali_1)
print(samples_known_pheno_quali.shape)
# Funciona correctamente

# Probemos brevemente la función compare_sample_ids
crude_id_comparison = integration.compare_crude_sample_ids(variants, crude_feno_quali_1)

print(
    f"{len(crude_id_comparison['shared'])} samples shared; "
    f"{len(crude_id_comparison['genotype_only'])} genotype-only; "
    f"{len(crude_id_comparison['phenotype_only'])} phenotype-only."
)

# ---------------------------------------------------------------------------------------------------------------------
# PROBEMOS AHORA EL FILTRADO DE DATOS GENOTÍPICOS

''''''
# Filtrado para el PCA y comprobamos cuántos SNPs e individuos quedan
filtered_vars_for_PCA = integration.filter_genotypes_for_PCA(variants, crude_feno_quali_2)

# matriz012_filt_PCA = pynei.pca.create_012_gt_matrix(filtered_vars_for_PCA, transform_to_biallelic=True)
# print(matriz012_filt_PCA.shape)
'''
#-------------------------------------------------------------------------------------
# Para el vcf de 20MB   --> (4122, 143)
# Para el vcf de 50MB   --> (8461, 143)
# Para el vcf de 100MB  --> (17334, 143)
#-------------------------------------------------------------------------------------
'''
sample_ids = filtered_vars_for_PCA.samples
# print(sample_ids)
''''''
# Filtrado para el GWAS y comprobamos cuántos SNPs e individuos quedan
filtered_vars_for_GWAS = integration.filter_genotypes_for_GWAS(variants, crude_feno_quali_2)

# matriz012_filt_GWAS = pynei.pca.create_012_gt_matrix(filtered_vars_for_GWAS, transform_to_biallelic=True)
# print(matriz012_filt_GWAS.shape)
'''
#-------------------------------------------------------------------------------------
# Para el vcf de 20MB   --> (23504, 143)
# Para el vcf de 50MB   --> (58979, 143)
# Para el vcf de 100MB  --> (117912, 143)
#-------------------------------------------------------------------------------------
'''
# print(filtered_vars_for_GWAS.samples)
''''''
# ¿Converge el PCA si solo filtramos por MAF y no por LD? EN ESTE CASO SÍ

#--------------------------------------------------------------------------------------------
# PROBEMOS LA FUNCIÓN do_gwas

# Primero, filtramos el pd.Series de fenotipos para que los individuos coincidan con el variants
filtered_phenotypes = integration.filter_phenotypes(crude_feno_quali_2, filtered_vars_for_GWAS.samples)
print(filtered_phenotypes.to_numpy().shape)

# Segundo, hacemos el PCA con Pynei, a partir del Variants filtrado
PCA = pynei.pca.do_pca_with_vars(filtered_vars_for_PCA, transform_to_biallelic=True)

# Ya tenemos todo
gwas_results = integration.do_gwas(filtered_vars_for_GWAS,
                filtered_phenotypes,
                PCA["projections"],
                type_of_phenotype="cualitativo (binario)",
                sort_by_significance=True
                )

print(gwas_results)