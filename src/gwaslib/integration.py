# Queremos crear una función general do_gwas
# Inputs: Genotipos: Variants, Fenotipos:pandas.df, parámetros_gwas)
# Parámetros son la MAF y el % de missing variants y missing samples
# 
#  
import pynei

# Identificar los individuos/muestras que cuyo % de datos faltantes es menor del umbral
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


# Función do_bonferroni(p-valores)

# Función do_gwas_quant(variants, phenotypes:pd.Dataframe?, covariables??: dataframe?, params)
# Función do_gwas_qual
'''