from pathlib import Path
import gc
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gwaslib as gw
import pynei

PROJECT_DIR = Path(__file__).parent.parent.parent
PCA_DIR = PROJECT_DIR / "geno_pheno_files" / "PCA_FILES_from_VCFs"

# ============================================================
# DATOS QUE CAMBIARÁS PARA CADA EJECUCIÓN
# ============================================================

path_VCF = PROJECT_DIR / "geno_pheno_files" / "VCF_FILES_from_Ximo" / "Varitome_reduced_all_chroms.vcf"
output_vcf_name = "100 MB"

path_PCs = PCA_DIR / "From_100mb_VCF" / "PCA_quanti_trait_mean_color_b.csv"
df_all_pcs = pd.read_csv(path_PCs, index_col=0)


path_phenotypes = PROJECT_DIR / "geno_pheno_files" / "PHENOTYPES_from_Ximo" / "CSV_files" / "quanti_trait_mean_color_b.csv"
phenotype_name = "Phenotype"

output_csv = Path(
    f"timing_{output_vcf_name.replace(' ', '_')}.csv"
)


# ============================================================
# PARÁMETROS
# ============================================================

N_REPEATS = 3

MAX_SAMPLE_MISSING = 0.05
MAX_SNP_MISSING = 0.05
MAX_MAJOR_ALLELE_FREQ = 0.95
MIN_R2 = 0.1

TYPE_OF_PHENOTYPE = "cuantitativo"


# ============================================================
# GUARDAR TIEMPOS
# ============================================================

times = {
    "load": [],
    "pca_filter": [],
    "pca": [],
    "gwas_filter": [],
    "gwas_graphs": [],
    "total": [],
}


# ============================================================
# REPETICIONES
# ============================================================

for repetition in range(1, N_REPEATS + 1):

    print(f"\n{'=' * 60}")
    print(f"REPETICIÓN {repetition}/{N_REPEATS} — {output_vcf_name}")
    print(f"{'=' * 60}")

    # --------------------------------------------------------
    # 1. CARGA DE FICHEROS
    # --------------------------------------------------------

    start = time.perf_counter()

    variants_crude = pynei.vars_from_vcf(path_VCF)
    phenotypes_crude = gw.load_phenotypes(path_phenotypes)

    t_load = time.perf_counter() - start
    times["load"].append(t_load)

    print(f"Carga de ficheros: {t_load:.2f} s")


    # --------------------------------------------------------
    # 2. FILTRADO PARA PCA
    # --------------------------------------------------------

    start = time.perf_counter()

    filtered_vars_for_pca = gw.filter_genotypes_for_PCA(
        variants=variants_crude,
        phenotypes=phenotypes_crude,
        max_sample_gt_missing_rate=MAX_SAMPLE_MISSING,
        max_var_gt_missing_rate=MAX_SNP_MISSING,
        max_allowed_maf=MAX_MAJOR_ALLELE_FREQ,
        min_allowed_r2=MIN_R2,
    )


    # Fuerza la ejecución efectiva del filtrado
    matrix012_pca = pynei.pca.create_012_gt_matrix(
        filtered_vars_for_pca,
        transform_to_biallelic=True,
    )

    t_pca_filter = time.perf_counter() - start
    times["pca_filter"].append(t_pca_filter)

    print(f"Filtrado para PCA: {t_pca_filter:.2f} s")


    # --------------------------------------------------------
    # 3. PCA
    # --------------------------------------------------------

    start = time.perf_counter()

    pca_input = pd.DataFrame(matrix012_pca.T)

    pca_results = pynei.pca.do_pca(pca_input)

    t_pca = time.perf_counter() - start
    times["pca"].append(t_pca)

    print(f"PCA: {t_pca:.2f} s")


    # --------------------------------------------------------
    # 4. FILTRADO PARA GWAS
    #    Medición independiente
    # --------------------------------------------------------

    start = time.perf_counter()

    filtered_vars_for_gwas_test = gw.filter_genotypes_for_GWAS(
        variants=variants_crude,
        phenotypes=phenotypes_crude,
        max_sample_gt_missing_rate=MAX_SAMPLE_MISSING,
        max_var_gt_missing_rate=MAX_SNP_MISSING,
        max_allowed_maf=MAX_MAJOR_ALLELE_FREQ,
    )

    # Fuerza la ejecución efectiva del filtrado
    matrix012_gwas = pynei.pca.create_012_gt_matrix(
        filtered_vars_for_gwas_test,
        transform_to_biallelic=True,
    )

    t_gwas_filter = time.perf_counter() - start
    times["gwas_filter"].append(t_gwas_filter)

    print(f"Filtrado para GWAS: {t_gwas_filter:.2f} s")

    # Ya no necesitamos esta matriz
    del matrix012_gwas
    del filtered_vars_for_gwas_test


    # --------------------------------------------------------
    # 5. GWAS + CREACIÓN DE LAS TRES GRÁFICAS
    # --------------------------------------------------------

    start = time.perf_counter()

    # Creamos un Variants nuevo para que do_gwas lo procese
    filtered_vars_for_gwas = gw.filter_genotypes_for_GWAS(
        variants=variants_crude,
        phenotypes=phenotypes_crude,
        max_sample_gt_missing_rate=MAX_SAMPLE_MISSING,
        max_var_gt_missing_rate=MAX_SNP_MISSING,
        max_allowed_maf=MAX_MAJOR_ALLELE_FREQ,
    )

    filtered_phenotypes = gw.filter_phenotypes(
        phenotypes_crude,
        filtered_vars_for_gwas.samples,
    )

    gwas_results = gw.do_gwas(
        filtered_vars=filtered_vars_for_gwas,
        filtered_phenotypes=filtered_phenotypes,
        covariates=df_all_pcs,
        type_of_phenotype=TYPE_OF_PHENOTYPE,
        sort_by_significance=False,
    )

    fig_pca, _ = gw.create_pca_plot(df_all_pcs)

    fig_manhattan, _ = gw.create_manhattan_plot(
        gwas_results,
        y_axis_variable="p",
        phenotype_name=phenotype_name,
    )

    fig_qq, _ = gw.create_qq_plot(
        gwas_results,
        phenotype_name=phenotype_name,
    )

    # No necesitamos guardar estas figuras durante el benchmark
    plt.close(fig_pca)
    plt.close(fig_manhattan)
    plt.close(fig_qq)

    t_gwas_graphs = time.perf_counter() - start
    times["gwas_graphs"].append(t_gwas_graphs)

    print(f"GWAS + gráficas: {t_gwas_graphs:.2f} s")


    # --------------------------------------------------------
    # TIEMPO TOTAL
    # --------------------------------------------------------
    t_total = (
        t_load
        + t_pca_filter
        + t_pca
        + t_gwas_filter
        + t_gwas_graphs
    )

    times["total"].append(t_total)

    print(f"Tiempo total: {t_total / 60:.2f} min")

    # Liberar memoria antes de la siguiente réplica
    del matrix012_pca
    del pca_results
    del gwas_results

    gc.collect()


# ============================================================
# MEDIA Y DESVIACIÓN ESTÁNDAR
# ============================================================

summary = {
    "VCF": output_vcf_name,
}

for stage in [
    "load",
    "pca_filter",
    "pca",
    "gwas_filter",
    "gwas_graphs",
    "total",
]:
    summary[f"{stage}_mean_s"] = np.mean(times[stage])
    summary[f"{stage}_sd_s"] = np.std(times[stage], ddof=1)


results_df = pd.DataFrame([summary])

results_df.to_csv(
    output_csv,
    index=False,
)

print("\n" + "=" * 60)
print("RESULTADOS")
print("=" * 60)
print(results_df.to_string(index=False))
print(f"\nGuardado en: {output_csv}")