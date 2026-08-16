'''
En este módulo crearemos funciones que permitan integrar las funciones de regresión de gwaslib
con las funciones de la librería pynei, necesarias para leer VCF, filtrar, hacer PCA, etc.
'''
from pathlib import Path
from io import BytesIO

from gwaslib.quantitative import linreg_3d, linreg_sm
from gwaslib.qualitative import logreg_3d, logreg_sm

import pynei
from pynei.gt_counts import _count_alleles_per_var
from pynei.config import DEF_POP_NAME
from pynei.variants import VariantsChunk

import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests

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

# Para identificar individuos que no coinciden entre el vcf y el excel/csv de fenotipos,
# antes de hacer ningún filtrado!
def compare_crude_sample_ids(variants:Variants, phenotypes: pd.Series) -> dict:
    genotype_samples = set(variants.samples)
    phenotype_samples = set(phenotypes.index)

    shared = genotype_samples & phenotype_samples
    genotype_only = genotype_samples - phenotype_samples
    phenotype_only = phenotype_samples - genotype_samples

    return {
        "shared": shared,
        "genotype_only": genotype_only,
        "phenotype_only": phenotype_only,
    }
'''
# Para visualizar esta información, correr algo como
crude_id_comparison = compare_crude_sample_ids(variants, phenotypes)

print(
    f"{len(crude_id_comparison['shared'])} samples shared; "
    f"{len(crude_id_comparison['genotype_only'])} genotype-only; "
    f"{len(crude_id_comparison['phenotype_only'])} phenotype-only."
)
'''
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
    
    # Por datos genotípicos faltantes

    # NOTA: Es muy importante hacer primero este filtro (antes del de NaN)
    # El motivo es que la función _get_samples_with_enough_genotype_data()
    # usa un objeto variants. Si hacemos primero el filtrado por NaN, el objeto variants
    # que le demos a esta función será de un solo uso, y lo consumiremos, impidiendo que sigamos filtrando

    samples_to_keep_geno = _get_samples_with_enough_genotype_data(
        variants, max_missing_rate=max_sample_gt_missing_rate # Aquí variants todavía no está filtrado, así que es reutilizable (no lo consumiremos)
    ) 
    variants = pynei.var_filters.filter_samples(variants, samples_to_keep_geno)

    # Por tener NaN en su fenotipo
    samples_to_keep_pheno = _get_samples_with_known_phenotypes(phenotypes)
    variants = pynei.var_filters.filter_samples(variants, samples_to_keep_pheno)


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
    
    # Por datos genotípicos faltantes

    # NOTA: Es muy importante hacer primero este filtro (antes del de NaN)
    # El motivo es que la función _get_samples_with_enough_genotype_data()
    # usa un objeto variants. Si hacemos primero el filtrado por NaN, el objeto variants
    # que le demos a esta función será de un solo uso, y lo consumiremos, impidiendo que sigamos filtrando

    samples_to_keep_geno = _get_samples_with_enough_genotype_data(
        variants, max_missing_rate=max_sample_gt_missing_rate # Aquí variants todavía no está filtrado, así que es reutilizable (no lo consumiremos)
    ) 
    variants = pynei.var_filters.filter_samples(variants, samples_to_keep_geno)

    # Por tener NaN en su fenotipo
    samples_to_keep_pheno = _get_samples_with_known_phenotypes(phenotypes)
    variants = pynei.var_filters.filter_samples(variants, samples_to_keep_pheno)


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


# Para extraer de un Chunk los datos necesarios para construir la tabla de resultados
def _extract_chunk_data(chunk: VariantsChunk) -> dict:
    """
    Extrae de un VariantsChunk:

    - matriz 012
    - cromosoma
    - posición
    - alelo mayoritario
    - alelos no mayoritarios

    La matriz 012 se calcula siguiendo la misma lógica utilizada por
    pynei.pca._create_012_gt_matrix(..., transform_to_biallelic=True).
    """

    # ============================================================
    # 1. Calcular alelo mayoritario de cada SNP
    # ============================================================

    res = _count_alleles_per_var(chunk, calc_freqs=False)

    allele_counts = (
        res["counts"][DEF_POP_NAME]["allele_counts"].values
    )

    num_genotyped_alleles_per_var = allele_counts.sum(axis=1)

    if np.any(num_genotyped_alleles_per_var == 0):
        raise ValueError(
            "There are variants that only have missing data"
        )

    major_allele_idxs = np.argmax(allele_counts, axis=1)

    # ============================================================
    # 2. Crear matriz 012
    # ============================================================

    gt_array = chunk.gts.gt_values

    matriz012 = np.sum(
        gt_array != major_allele_idxs[:, None, None],
        axis=2,
    )

    # ============================================================
    # 3. Obtener metadata de los SNPs
    # ============================================================

    chromosomes = chunk.vars_info["chrom"].to_numpy()
    positions = chunk.vars_info["pos"].to_numpy()

    major_alleles = []
    non_major_alleles = []

    for variant_idx, major_idx in enumerate(major_allele_idxs):

        alleles = chunk.alleles.iloc[variant_idx]

        major_allele = alleles.iloc[major_idx]

        other_alleles = [
            allele
            for idx, allele in enumerate(alleles)
            if idx != major_idx and not pd.isna(allele) # Pynei rellena con <NA> las columnas de alelos que no existen
]
        # Añadimos ambos a las listas
        major_alleles.append(major_allele)

        # Lo dejamos como string para que el DataFrame final sea sencillo.
        # En variantes multialélicas puede contener varios alelos.
        non_major_alleles.append(",".join(other_alleles))

    return {
        "mat012": matriz012,
        "chromosome": chromosomes,
        "position": positions,
        "major_allele": np.asarray(major_alleles),
        "non_major_alleles": np.asarray(non_major_alleles),
    }


# Función do_gwas(filtered_variants, phenotypes:pd.Series, covariables??: dataframe?, type_of_phenotype)
def do_gwas(filtered_vars:Variants, 
        filtered_phenotypes:pd.Series, 
        covariates:pd.DataFrame,
        type_of_phenotype: "cuantitativo" | "cualitativo (binario)",
        sort_by_significance: bool = False,
        desired_chunk_size:int=10_000,
        ):

    # Comprobar si los IDs de todas las muestras coinciden en orden entre genotypes, phenotypes y covariates
    sample_ids = filtered_vars.samples

    if not np.array_equal(sample_ids, filtered_phenotypes.index.to_numpy()):
        raise ValueError(
            "The sample IDs in the phenotype table do not match the filtered genotype data."
        )

    if not np.array_equal(sample_ids, covariates.index.to_numpy()):
        raise ValueError(
            "The sample IDs in the covariate table do not match the filtered genotype data."
        )
    # ============================================================
    # 2. Pasar phenotype y covariables a NumPy
    # ============================================================

    phenotypes_array = filtered_phenotypes.to_numpy(dtype=float)

    covariates_array = covariates.to_numpy(dtype=float)

    # Coger solo 10 PCs
    if covariates_array.shape[1] > 10:
        covariates_array = covariates_array[:, :10]

    # ============================================================
    # 3. Elegir función de GWAS
    # ============================================================

    if type_of_phenotype == "cuantitativo":
        regression_function = linreg_3d

    elif type_of_phenotype == "cualitativo (binario)":
        regression_function = logreg_3d

    else:
        raise ValueError(
            "phenotype_type must be 'quantitative' or 'qualitative'."
        )

    # ============================================================
    # 4. Listas donde iremos acumulando resultados de los chunks
    # ============================================================

    chromosomes = []
    positions = []
    major_alleles = []
    non_major_alleles = []

    betas = []
    ses = []
    p_values = []

    # ============================================================
    # 5. Procesamiento chunk por chunk
    # ============================================================

    for chunk in filtered_vars.iter_vars_chunks(
        desired_num_vars_per_chunk=desired_chunk_size
    ):

        # -----------------------------------------
        # Extraer genotipos + metadata
        # -----------------------------------------

        chunk_data = _extract_chunk_data(chunk)

        matriz012_chunk = chunk_data["mat012"]

        # -----------------------------------------
        # Ejecutar regresiones del chunk
        # -----------------------------------------
        
        results = regression_function(
            matriz012_chunk,
            phenotypes_array,
            covariates_array,
        )

        # -----------------------------------------
        # Guardar metadata
        # -----------------------------------------

        chromosomes.append(
            chunk_data["chromosome"]
        )

        positions.append(
            chunk_data["position"]
        )

        major_alleles.append(
            chunk_data["major_allele"]
        )

        non_major_alleles.append(
            chunk_data["non_major_alleles"]
        )

        # -----------------------------------------
        # Guardar resultados del GWAS
        # -----------------------------------------
        
        betas.append(results["beta"])
        ses.append(results["SE"])
        p_values.append(results["p_val"])

    # ============================================================
    # 6. Unir resultados de todos los chunks
    # ============================================================

    chromosomes = np.concatenate(chromosomes)
    positions = np.concatenate(positions)
    major_alleles = np.concatenate(major_alleles)
    non_major_alleles = np.concatenate(non_major_alleles)

    betas = np.concatenate(betas)
    ses = np.concatenate(ses)
    p_values = np.concatenate(p_values)

    # ============================================================
    # 7. Corrección por múltiples tests
    # ============================================================
    p_bonferroni = multipletests(p_values, method="bonferroni")[1]
    p_fdr = multipletests(p_values, method="fdr_bh")[1]

    # ============================================================
    # 8. -log10
    # ============================================================

    # Evita log10(0) si algún p-valor es tan pequeño que NumPy
    # lo representa como cero.
    tiny = np.finfo(float).tiny

    neg_log10_p = -np.log10(np.maximum(p_values, tiny))

    neg_log10_bonferroni = -np.log10(np.maximum(p_bonferroni, tiny))

    neg_log10_fdr = -np.log10(np.maximum(p_fdr, tiny))

    # ============================================================
    # 9. DataFrame final
    # ============================================================
    gwas_results = pd.DataFrame(
        {
            "Chromosome": chromosomes,
            "Position": positions,
            "Non-effect allele": major_alleles,
            "Effect allele(s)": non_major_alleles,
            "beta": betas,
            "SE": ses,
            "-log10(p)": neg_log10_p,
            "-log10(Bonferroni)": neg_log10_bonferroni,
            "-log10(FDR_BH)": neg_log10_fdr,
        }
    )

    if sort_by_significance:
        gwas_results = gwas_results.sort_values(
            by="-log10(p)",
            ascending=False,
        ).reset_index(drop=True)

    return gwas_results