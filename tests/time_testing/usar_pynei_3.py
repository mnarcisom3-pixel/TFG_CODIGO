from pathlib import Path

from gwaslib import integration, visualization
import pynei

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).parent.parent.parent

# SELECCIONAR VCF ------------------------------------------------------------------------------------------
# Una versión aún más reducida (20MB) para que el tiempo de computación sea menor
#path_VCF = PROJECT_DIR / "geno_pheno_files" / "VCF_FILES_from_Ximo" / "Varitome_20mb_reduced.vcf"

# Una versión intermedia (50MB)
#path_VCF = PROJECT_DIR / "geno_pheno_files" / "VCF_FILES_from_Ximo" / "Varitome_50mb_reduced.vcf"

# El de probar la web, de Varitome reducido a 100MB
path_VCF = PROJECT_DIR / "geno_pheno_files" / "VCF_FILES_from_Ximo" / "Varitome_reduced_all_chroms.vcf"


# LEER VCF CON PYNEI ----------------------------------------------------------------------------------------
variants = pynei.io_vcf.vars_from_vcf(path_VCF)

matriz012_crude = pynei.pca.create_012_gt_matrix(variants, transform_to_biallelic=True)
print(matriz012_crude.shape)

# LEER FICHERO DE FENOTIPOS ---------------------------------------------------------------------------------
# Cuantitativos
#path_excel_quanti = PROJECT_DIR / "geno_pheno_files" / "PHENOTYPES_from_Ximo" / "Excel_files" / "quanti_trait_mean_fruit_weight.xlsx"
path_csv_quanti = PROJECT_DIR / "geno_pheno_files" / "PHENOTYPES_from_Ximo" / "CSV_files" / "quanti_trait_mean_color_b.csv"

df_crude_feno_quanti = integration.load_phenotypes(path_csv_quanti)

# Comparación de muestras VCF-Fenotipos
crude_id_comparison = integration.compare_crude_sample_ids(variants, df_crude_feno_quanti)

print(
    f"{len(crude_id_comparison['shared'])} samples shared; "
    f"{len(crude_id_comparison['genotype_only'])} genotype-only; "
    f"{len(crude_id_comparison['phenotype_only'])} phenotype-only."
)

# FILTRAR GENOTIPOS Y FENOTIPOS PARA EL GWAS
filtered_vars_for_GWAS = integration.filter_genotypes_for_GWAS(variants, df_crude_feno_quanti)

df_filtered_feno = integration.filter_phenotypes(df_crude_feno_quanti, filtered_vars_for_GWAS.samples)
print(df_filtered_feno.to_numpy().shape)

# PCA
'''
# El filtrado para el PCA lo omitimos para ahorrar tiempo. 
# Importaremos el PCA ya calculado desde un csv
filtered_vars_for_PCA = integration.filter_genotypes_for_PCA(variants, df_crude_feno_quanti)
'''
# Importamos CSV
PCA_DIR = PROJECT_DIR / "geno_pheno_files" / "PCA_FILES_from_VCFs"
path_PCA = PCA_DIR / "From_100mb_VCF" / "PCA_quanti_trait_mean_color_b.csv"


df_all_pcs = pd.read_csv(path_PCA, index_col=0)

# PCA plot
fig, ax = visualization.create_pca_plot(df_all_pcs)
plt.show()

# EJECUTAR GWAS ----------------------------------------------------------------------------------------------
gwas_results = integration.do_gwas(filtered_vars=filtered_vars_for_GWAS,
                filtered_phenotypes=df_filtered_feno,
                covariates=df_all_pcs,
                type_of_phenotype="cuantitativo",
                sort_by_significance=False,
                )

print(gwas_results)

# Manhattan plot
fig, ax = visualization.create_manhattan_plot(
    gwas_results,
    y_axis_variable="p",
    phenotype_name="Mean Color b"
)
plt.show()

# QQ-plot
fig, ax = visualization.create_qq_plot(
    gwas_results,
    phenotype_name="Mean Color b",
)
plt.show()
