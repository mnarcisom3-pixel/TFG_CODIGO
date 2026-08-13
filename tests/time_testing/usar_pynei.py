import numpy as np

from pathlib import Path

from gwaslib import integration
import pynei
PROJECT_DIR = Path(__file__).parent.parent.parent

path_VCF = PROJECT_DIR / "geno_pheno_files" / "vcf_test_247rows.vcf"
# path_VCF = PROJECT_DIR / "geno_pheno_files" / "vcf_7_samples.vcf"
# 

# Leemos el VCF con pynei
variants = pynei.io_vcf.vars_from_vcf(path_VCF)
matriz012_sin = pynei.pca.create_012_gt_matrix(variants, transform_to_biallelic=True)
print(matriz012_sin.shape)

'''
# DUDA 1 = Vamos a comprobar si un objeto Variants (tal cual del VCF) es un iterador de un solo uso
matriz012_sin_2 = pynei.pca.create_012_gt_matrix(variants, transform_to_biallelic=True)
print(matriz012_sin_2.shape)

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
# DUDA 3 = ¿¿Obtener las samples_idx de un Variants filtrado consumirá el iterador??
# Probemos
filt = pynei.var_filters.filter_by_ld_and_maf(variants)
a = filt.samples
print(a)
b = filt.samples
print(b)
# No lo consume


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
'''
variants_2 = pynei.io_vcf.vars_from_vcf(path_VCF)
filtered_vars_for_PCA = integration.filter_genotypes_for_PCA(variants_2)
'''
'''
for chunk in filtered_vars_2.iter_vars_chunks():
    print(chunk.alleles.head())
    break
'''
'''
matriz012_filt_2 = pynei.pca.create_012_gt_matrix(filtered_vars_for_PCA, transform_to_biallelic=True)
print(matriz012_filt_2.shape)

sample_ids = filtered_vars_for_PCA.samples
print(sample_ids.shape)


variants_3 = pynei.io_vcf.vars_from_vcf(path_VCF)
filtered_vars_for_GWAS = integration.filter_genotypes_for_GWAS(variants_3)
'''
'''
matriz012_filt_3 = pynei.pca.create_012_gt_matrix(filtered_vars_for_GWAS, transform_to_biallelic=True)
print(matriz012_filt_3.shape)
'''
# ¿Converge el PCA si solo filtramos por MAF y no por LD? EN ESTE CASO SÍ

'''
# Sigamos probando funciones de integration.py
all_info = integration._extract_variant_data(filtered_vars_for_GWAS)
print(all_info.keys())
'''

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
