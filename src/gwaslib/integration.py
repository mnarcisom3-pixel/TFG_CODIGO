'''
En este módulo crearemos funciones que permitan integrar las funciones de regresión de gwaslib
con las funciones de la librería pynei, necesarias para leer VCF, filtrar, hacer PCA, etc.
'''
from pathlib import Path
from io import BytesIO

import pynei
import gwaslib as gw

import pandas as pd
import numpy as np

# Leer el archivo input de fenotipos (.csv o .xlsx)
# Esta función hace, básicamente, 4 cosas: 
#   - Leer archivo .csv o .xlsx
#   - Verificar las exigencias de contenido
#   - Convertir a valores numéricos (comas decimales por puntos)
#   - Crear un pd.Series para usarlo en do_gwas()
def load_phenotypes(
    source: str | Path | bytes,
    filename: str | None = None,
) -> pd.Series:
    """
    Carga los datos fenotípicos de un Excel o un CSV.
    El archivo input debe contener exactamente 2 columnas, en este orden
    'Sample' y 'Phenotype'.

    Parameters
    ----------
    source : str | Path | bytes
        Local file path or raw bytes from an uploaded file.

    filename : str | None, optional
        Original filename. Required when `source` is bytes,
        in order to determine the file format.

    Returns
    -------
    pd.Series
        Phenotype values indexed by sample ID.
    """

    # =========================================================
    # 1. Determine the input source and file extension
    # =========================================================

    if isinstance(source, (str, Path)):
        file_source = Path(source)
        suffix = file_source.suffix.lower()

    elif isinstance(source, bytes):
        if filename is None:
            raise ValueError(
                "'filename' must be provided when 'source' is bytes."
            )

        file_source = BytesIO(source)
        suffix = Path(filename).suffix.lower()

    else:
        raise TypeError(
            "'source' must be a str, pathlib.Path, or bytes object."
        )

    # =========================================================
    # 2. Read file
    # =========================================================

    if suffix == ".csv":

        # sep=None allows Pandas to detect delimiters such as
        # ',' or ';' automatically.
        df = pd.read_csv(
            file_source,
            sep=None,
            engine="python",
        )

    elif suffix == ".xlsx":

        # calamine works in Pyodide
        df = pd.read_excel(
            file_source,
            engine="calamine",
        )

    else:
        raise ValueError(
            "Phenotype file must have '.csv' or '.xlsx' extension."
        )

    # =========================================================
    # 3. Validate table structure
    # =========================================================

    expected_columns = ["Sample", "Phenotype"]

    if list(df.columns) != expected_columns:
        raise ValueError(
            "Phenotype file must contain exactly two columns, "
            "in this order: 'Sample' and 'Phenotype'."
        )

    # =========================================================
    # 4. Validate sample IDs
    # =========================================================

    if df["Sample"].isna().any():
        raise ValueError(
            "Missing sample IDs were found in the phenotype file."
        )

    if df["Sample"].duplicated().any():
        raise ValueError(
            "Duplicate sample IDs were found in the phenotype file."
        )

    # =========================================================
    # 5. Normalize phenotype values
    # =========================================================

    phenotype_raw = df["Phenotype"]

    # Keep track of genuine missing values.
    missing = phenotype_raw.isna()

    # This is to make both 12.34 and 12,34 valid decimal representations.
    normalized = (
        phenotype_raw.astype("string")
        .str.strip()
        .str.replace(",", ".", regex=False)
    )

    phenotype_numeric = pd.to_numeric(
        normalized,
        errors="coerce", # Para no sacar error por pantalla cuando un valor no pueda convertirse a número (pasa a NaN))
    )

    # Detect values which were NOT missing originally but could not
    # be converted to numbers.
    invalid = (~missing) & phenotype_numeric.isna()

    if invalid.any():
        invalid_values = phenotype_raw[invalid].unique()

        raise ValueError(
            "Non-numeric phenotype values were found: "
            + ", ".join(map(str, invalid_values))
        )

    df["Phenotype"] = phenotype_numeric

    # =========================================================
    # 6. Convert to Series
    # =========================================================

    phenotypes = df.set_index("Sample")["Phenotype"]

    return phenotypes

# Para identificar los individuos/muestras cuyo fenotipo es conocido (no es NaN)
def _get_samples_with_known_phenotypes(phenotypes: pd.Series):
    return phenotypes.index[phenotypes.notna()].to_numpy()


# Para identificar los individuos/muestras cuyo % de datos faltantes es menor del umbral
def _get_samples_with_enough_genotype_data(variants: Variants, max_missing_rate):
    sample_stats = pynei.calc_per_sample_stats(variants)
    samples_with_enough_data = tuple(
        sorted(
            (sample_stats.index[sample_stats["missing_gt_rate"] <= max_missing_rate])
        )
    )
    return samples_with_enough_data

# Para filtrar datos para PCA según los parámetros indicados
def filter_genotypes_for_PCA(variants: Variants,
        phenotypes:pd.Series,
        max_sample_gt_missing_rate=0.05,
        max_var_gt_missing_rate=0.05,
        max_allowed_maf=0.95,
        min_allowed_r2=0.1,
        ):
    
    # Filtrado de muestras/individuos
    # Esto debe hacerse primero, ya que los individuos afectan al cálculo de MAF y del %missingness de SNP
    
    # Por tener NaN en su fenotipo
    samples_to_keep_pheno = _get_samples_with_known_phenotypes(phenotypes)
    variants = pynei.var_filters.filter_samples(variants, samples_to_keep_pheno)

    # Por datos genotípicos faltantes
    samples_to_keep_geno = _get_samples_with_enough_genotype_data(
        variants, max_missing_rate=max_sample_gt_missing_rate
    )
    variants = pynei.var_filters.filter_samples(variants, samples_to_keep_geno)

    # Filtrado de SNPs, por datos faltantes, LD y MAF
    variants = pynei.filter_by_missing_data(
        variants, max_allowed_missing_rate=max_var_gt_missing_rate
    )

    variants = pynei.filter_by_ld_and_maf(
        variants, max_allowed_maf=max_allowed_maf, min_allowed_r2=min_allowed_r2
    )
    return variants

# Para filtrar datos para GWAS según los parámetros indicados
def filter_genotypes_for_GWAS(variants: Variants,
        phenotypes:pd.Series,
        max_sample_gt_missing_rate=0.05,
        max_var_gt_missing_rate=0.05,
        max_allowed_maf=0.95,
        ):

    # Filtrado de muestras/individuos
    # Esto debe hacerse primero, ya que los individuos afectan al cálculo de MAF y del %missingness de SNP
    
    # Por tener NaN en su fenotipo
    samples_to_keep_pheno = _get_samples_with_known_phenotypes(phenotypes)
    variants = pynei.var_filters.filter_samples(variants, samples_to_keep_pheno)

    # Por datos genotípicos faltantes
    samples_to_keep_geno = _get_samples_with_enough_genotype_data(
        variants, max_missing_rate=max_sample_gt_missing_rate
    )
    variants = pynei.var_filters.filter_samples(variants, samples_to_keep_geno)

    # Filtrado de SNPs, por datos faltantes y MAF (para GWAS no filtramos por LD)
    variants = pynei.filter_by_missing_data(
        variants, max_allowed_missing_rate=max_var_gt_missing_rate
    )
    variants = pynei.filter_by_maf(variants, max_allowed_maf=max_allowed_maf)

    return variants


# Para filtrar individuos en el pd.Series de fenotipos
def filter_phenotypes(
    phenotypes: pd.Series,
    filtered_samples_idx,
) -> pd.Series:
    """
    Selecciona y reordena los fenotipos de los individuos de filtered_samples_idx
    (Los que pasaron la criba al filtrar el objeto Variants)
    
    Parámetros
    ----------
    phenotypes : pd.Series
        Valores de fenotipos, indexados por ID de muestra

    filtered_samples_idx
        IDs de muestra que deseamos mantener, en el orden deseado

    Returns
    -------
    pd.Series
        Fenotipos filtrados, ordenados igual que filtered_samples_idx

    Raises
    ------
    ValueError
        Si hay alguna muestra de filtered_samples_idx que no exista en phenotypes
    """
    missing_samples = [
        sample for sample in filtered_samples_idx
        if sample not in phenotypes.index
    ]

    if missing_samples:
        raise ValueError(
            "Some samples present in the genotype data are missing "
            "from the phenotype data: "
            + ", ".join(map(str, missing_samples))
        )

    return phenotypes.loc[filtered_samples_idx]


# Para extraer datos necesarios para construir la tabla de resultados
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


# Función do_gwas(filtered_variants, phenotypes:pd.Series, covariables??: dataframe?, type_of_phenotype)

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
        
