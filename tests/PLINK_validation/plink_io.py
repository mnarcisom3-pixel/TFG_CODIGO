"""
Funciones de I/O compartidas para exportar datos de genotipo/fenotipo/covariables
al formato que necesita PLINK2. Se usa tanto desde export_quanti_forplink.py como desde
export_quali_forplink.py.
"""

import numpy as np
import pandas as pd


def export_to_plink(matriz012, phenotypes, covariables, sample_ids, out_prefix):
    """
    matriz012    : (M_snps, N_indivs) dosis alélica 0/1/2
    phenotypes   : (N_indivs,) fenotipo (cuantitativo o binario 0/1)
    covariables  : (N_indivs, K) covariables (p.ej. PCs), o None si no se quieren exportar
    sample_ids   : lista de N_indivs identificadores de muestra (str)
    out_prefix   : prefijo de los ficheros de salida

    Convención: dosis 0 -> 0/0 (hom. REF), 1 -> 0/1 (het), 2 -> 1/1 (hom. ALT).
    REF/ALT son arbitrarios (no hay alelos "reales"), así que conviene comprobar
    en la salida de PLINK qué alelo ha quedado como A1 (ver compare_gwas_*.py).
    """
    M, N = matriz012.shape
    assert len(sample_ids) == N
    assert len(phenotypes) == N

    # --- VCF ---
    geno_str = np.empty(matriz012.shape, dtype=object)
    geno_str[matriz012 == 0] = "0/0"
    geno_str[matriz012 == 1] = "0/1"
    geno_str[matriz012 == 2] = "1/1"

    with open(f"{out_prefix}.vcf", "w") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        f.write(
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
            + "\t".join(sample_ids)
            + "\n"
        )
        for i in range(M):
            row = geno_str[i, :]
            f.write(
                f"1\t{i + 1}\tSNP_{i:05d}\tA\tT\t.\tPASS\t.\tGT\t"
                + "\t".join(row)
                + "\n"
            )

    # --- Fenotipo ---
    pheno_df = pd.DataFrame({"#FID": sample_ids, "IID": sample_ids, "PHENO": phenotypes})
    pheno_df.to_csv(f"{out_prefix}_pheno.txt", sep="\t", index=False)

    # --- Covariables (PCs), opcional ---
    if covariables is not None:
        assert covariables.shape[0] == N
        cov_cols = {f"PC{i + 1}": covariables[:, i] for i in range(covariables.shape[1])}
        covar_df = pd.DataFrame({"#FID": sample_ids, "IID": sample_ids, **cov_cols})
        covar_df.to_csv(f"{out_prefix}_own_covars.txt", sep="\t", index=False)

    print(f"Generados los ficheros con prefijo '{out_prefix}'")

# Función para exportar a VCF los individuos y SNPs filtrados por Pynei (a partir de datos reales)
def export_to_plink_with_real_positions(
    matriz012,
    phenotypes,
    covariables,
    sample_ids,
    gwas_results,
    out_prefix,
):
    """
    Exporta los datos a PLINK2 usando:

    - genotipos reales codificados como 0/1/2;
    - cromosomas y posiciones reales;
    - alelos artificiales A/T;
    - IDs artificiales SNP_00000, SNP_00001, ...

    IMPORTANTE:
    gwas_results debe estar en el mismo orden que las filas de matriz012
    (es decir, do_gwas con sort_by_significance=False).
    """

    M, N = matriz012.shape

    assert len(sample_ids) == N
    assert len(phenotypes) == N
    assert len(gwas_results) == M

    # ============================================================
    # VCF
    # ============================================================

    geno_str = np.empty(matriz012.shape, dtype=object)

    geno_str[matriz012 == 0] = "0/0"
    geno_str[matriz012 == 1] = "0/1"
    geno_str[matriz012 == 2] = "1/1"

    with open(f"{out_prefix}.vcf", "w") as f:

        f.write("##fileformat=VCFv4.2\n")

        f.write(
            '##FORMAT=<ID=GT,Number=1,Type=String,'
            'Description="Genotype">\n'
        )

        f.write(
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
            + "\t".join(sample_ids)
            + "\n"
        )

        for i in range(M):

            # Cromosoma original, por ejemplo:
            # SL2.50ch01 -> 1
            chromosome_original = str(
                gwas_results.iloc[i]["Chromosome"]
            )

            chromosome = int(
                chromosome_original[-2:]
            )

            position = int(
                gwas_results.iloc[i]["Position"]
            )

            snp_id = f"SNP_{i:05d}"

            row = geno_str[i, :]

            f.write(
                f"{chromosome}\t"
                f"{position}\t"
                f"{snp_id}\t"
                f"A\t"
                f"T\t"
                f".\t"
                f"PASS\t"
                f".\t"
                f"GT\t"
                + "\t".join(row)
                + "\n"
            )

    # ============================================================
    # Fenotipo
    # ============================================================

    pheno_df = pd.DataFrame(
        {
            "#FID": sample_ids,
            "IID": sample_ids,
            "PHENO": phenotypes,
        }
    )

    pheno_df.to_csv(
        f"{out_prefix}_pheno.txt",
        sep="\t",
        index=False,
    )

    # ============================================================
    # Covariables
    # ============================================================

    if covariables is not None:

        assert covariables.shape[0] == N

        cov_cols = {
            f"PC{i + 1}": covariables[:, i]
            for i in range(covariables.shape[1])
        }

        covar_df = pd.DataFrame(
            {
                "#FID": sample_ids,
                "IID": sample_ids,
                **cov_cols,
            }
        )

        covar_df.to_csv(
            f"{out_prefix}_own_covars.txt",
            sep="\t",
            index=False,
        )

    print(
        f"Generados los ficheros con prefijo '{out_prefix}'"
    )