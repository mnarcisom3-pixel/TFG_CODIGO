import pynei
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.parent

path_VCF = PROJECT_DIR / "geno_pheno_files" / "vcf_test_247rows.vcf"
# path_VCF = PROJECT_DIR / "geno_pheno_files" / "vcf_7_samples.vcf"
# 

# Leemos el VCF con pynei
variants = pynei.io_vcf.vars_from_vcf(path_VCF)

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
