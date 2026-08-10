'''
En este módulo crearemos funciones que permitan integrar las funciones de regresión de gwaslib
con las funciones de la librería pynei, necesarias para leer VCF, filtrar, hacer PCA, etc.
'''

import pynei
import gwaslib as gw

import pandas as pd
import numpy as np

# Identificar los individuos/muestras cuyo % de datos faltantes es menor del umbral
def _identify_samples_with_enough_data(variants: Variants, max_missing_rate):
    sample_stats = pynei.calc_per_sample_stats(variants)
    samples_with_enough_data = tuple(
        sorted(
            (sample_stats.index[sample_stats["missing_gt_rate"] <= max_missing_rate])
        )
    )
    return samples_with_enough_data

# Función "filter_data_for_PCA"(variants, params)
def filter_data_for_PCA(variants: Variants,
        max_sample_gt_missing_rate=0.05,
        max_var_gt_missing_rate=0.05,
        max_allowed_maf=0.95,
        min_allowed_r2=0.1,
        ):
    samples_to_keep = _identify_samples_with_enough_data(
        variants, max_missing_rate=max_sample_gt_missing_rate
    )
    # Filtramos por datos faltantes de muestra y de SNP
    variants = pynei.var_filters.filter_samples(variants, samples_to_keep)
    variants = pynei.filter_by_missing_data(
        variants, max_allowed_missing_rate=max_var_gt_missing_rate
    )
    # Filtramos por LD y MAF
    variants = pynei.filter_by_ld_and_maf(
        variants, max_allowed_maf=max_allowed_maf, min_allowed_r2=min_allowed_r2
    )
    return variants

# Función "filter_data_for_GWAS"(variants, params)
def filter_data_for_GWAS(variants: Variants,
        max_sample_gt_missing_rate=0.05,
        max_var_gt_missing_rate=0.05,
        max_allowed_maf=0.95,
        ):
    samples_to_keep = _identify_samples_with_enough_data(
        variants, max_missing_rate=max_sample_gt_missing_rate
    )
    # Filtramos por datos faltantes de muestra y de SNP
    variants = pynei.var_filters.filter_samples(variants, samples_to_keep)
    variants = pynei.filter_by_missing_data(
        variants, max_allowed_missing_rate=max_var_gt_missing_rate
    )
    # Filtramos por MAF
    variants = pynei.filter_by_maf(variants, max_allowed_maf=max_allowed_maf)
    return variants

'''
# Función para hacer el PCA con los parámetros de filtrado indicados
# Inputs --> Objeto variants, 4 parámetros
def do_pca_with_filters(variants:Variants,
        max_sample_gt_missing_rate=0.05,
        max_var_gt_missing_rate=0.05,
        max_allowed_maf=0.95,
        min_allowed_r2=0.1,
        ):

    filtered_vars = filter_data_for_PCA(variants, 
            max_sample_gt_missing_rate,
            max_var_gt_missing_rate,
            max_allowed_maf,
            min_allowed_r2
            )

    pca = pynei.do_pca_with_vars(filtered_vars, transform_to_biallelic=True)
    return pca
'''


# Función para extraer datos necesarios para construir la tabla de resultados
def _extract_variant_data(filtered_vars:Variants):
    """
    Extrae toda la información necesaria de un objeto Variants ya filtrado
    realizando únicamente una pasada sobre los chunks.

    Returns
    -------
    dict
        {
            "mat012": np.ndarray,
            "chromosome": list[str],
            "position": list[int],
            "major_allele": list[str],
            "non_major_alleles": list[str]
        }
    """

    mat012_chunks = []
    chromosomes = []
    positions = []
    major_alleles_list = []
    non_major_alleles_list = []

    for chunk in filtered_vars.iter_vars_chunks():

        # ==========================================================
        # Construcción de la matriz012
        # (idéntica a la implementación de Pynei)
        # ==========================================================

        res = pynei.gt_counts._count_alleles_per_var(chunk, calc_freqs=False)

        allele_counts = (
            res["counts"][pynei.config.DEF_POP_NAME]["allele_counts"].values
        )

        num_genotyped_alleles_per_var = allele_counts.sum(axis=1)

        if np.any(num_genotyped_alleles_per_var == 0):
            raise ValueError(
                "There are variants that only have missing data."
            )

        major_allele_idxs = np.argmax(allele_counts, axis=1)

        gt_array = chunk.gts.gt_values

        gts012 = np.sum(
            gt_array != major_allele_idxs[:, None, None],
            axis=2,
        )

        mat012_chunks.append(gts012)

        # ==========================================================
        # Información de los SNPs
        # ==========================================================

        chromosomes.extend(chunk.vars_info["chrom"].tolist())
        positions.extend(chunk.vars_info["pos"].tolist())

        allele_table = chunk.alleles.to_numpy()

        for row, major_idx in enumerate(major_allele_idxs):

            alleles = allele_table[row]

            major_alleles_list.append(alleles[major_idx])

            non_major = []

            for idx, allele in enumerate(alleles):

                if idx == major_idx:
                    continue

                if pd.isna(allele):
                    continue

                non_major.append(str(allele))

            non_major_alleles_list.append(",".join(non_major))

    mat012 = np.vstack(mat012_chunks)

    return {
        "mat012": mat012,
        "chromosome": chromosomes,
        "position": positions,
        "major_allele": major_alleles_list,
        "non_major_alleles": non_major_alleles_list,
    }

# Función do_gwas(filtered_variants, phenotypes:pd.Dataframe?, covariables??: dataframe?, type_of_phenotype)

def do_gwas(filtered_vars:Variants, 
        phenotypes:pd.Series, 
        covariates:pd.DataFrame,
        type_of_phenotype: "cuantitativo" | "cualitativo (binario)",
        ):

    # Comprobar si los IDs de todas las muestras coinciden en orden entre genotypes, phenotypes y covariates
    sample_ids = filtered_vars.samples

    if not np.array_equal(sample_ids, phenotypes.index.to_numpy()):
        raise ValueError(
            "The sample IDs in the phenotype table do not match the filtered genotype data."
        )

    if not np.array_equal(sample_ids, covariates.index.to_numpy()):
        raise ValueError(
            "The sample IDs in the covariate table do not match the filtered genotype data."
        )

    # Extraer toda la info. necesaria del objeto Variants filtrado
    all_info = _extract_variant_data(filtered_vars)

    mat_012 = all_info['mat012']
    chromosomes = all_info['chromosome']
    positions = all_info['position']
    major_alleles = all_info['major_allele']
    non_major_alleles = all_info['non_major_alleles']

    # A partir del pd.Series de fenotipos, obtener un np.ndarray
    phenotypes_array = phenotypes.to_numpy()
    # A partir del pd.Dataframe de covariables, obtener un np.ndarray
    covariates_array = covariates.to_numpy()
    # Si K es mayor que 10, coger solo los 10 primeros PCs
    if covariates_array.shape[1] > 10:
        covariates_array = covariates_array[:, :10]

    # Si el usuario indicó que desea hacer un GWAS cuantitativo
    if type_of_phenotype=='cuantitativo':
        res = gw.quantitative.linreg_3d(mat_012, phenotypes_array, covariates_array)


    # Si, en cambio, indicó que desea hacer un cualitativo
    elif type_of_phenotype=='cualitativo (binario)':
        res = gw.qualitative.logreg_3d(mat_012, phenotypes_array, covariates_array)

    # Construir el Dataframe de resultados
    # Extraer cromosoma, posición, y alelos de cada SNP

    # Calcular -log10(p-valores)

    # Calcular p-valores corregidos por Bonferroni y FDR

    # Construir Dataframe con todos los SNPs ordenados por cromosoma y posición
        
