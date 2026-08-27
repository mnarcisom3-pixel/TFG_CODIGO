import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def create_manhattan_plot(
    gwas_results: pd.DataFrame,
    phenotype_name: str,
    y_axis_variable: str = "p",
    alpha: float = 0.05,
    show_significance_threshold: bool = True,
):
    """
    Crea un Manhattan plot a partir del DataFrame generado por do_gwas.

    Parameters
    ----------
    gwas_results : pd.DataFrame
        DataFrame generado por do_gwas, sin ordenar por significancia.

    phenotype_name : str
        Nombre del carácter/fenotipo analizado.

    y_axis_variable : str, default="p"
        Variable representada en el eje Y:

        - "p": -log10(p)
        - "bonferroni": -log10(p ajustado por Bonferroni)
        - "fdr": -log10(p ajustado por Benjamini-Hochberg)

    show_significance_threshold : bool, default=True
        Si True, muestra una línea horizontal correspondiente
        a un nivel de significancia alpha indicado, así como el corregido por múltiples tests.

    Returns
    -------
    fig, ax
        Figura y eje de Matplotlib.
    """

    # ============================================================
    # 1. Seleccionar variable del eje Y
    # ============================================================

    y_columns = {
        "p": "-log10(p)",
        "bonferroni": "-log10(Bonferroni)",
        "fdr": "-log10(FDR_BH)",
    }

    if y_axis_variable not in y_columns:
        raise ValueError(
            "y_axis_variable debe ser 'p', 'bonferroni' o 'fdr'."
        )

    y_column = y_columns[y_axis_variable]

    # ============================================================
    # 2. Comprobar columnas necesarias
    # ============================================================

    required_columns = {
        "Chromosome",
        "Position",
        y_column,
    }

    missing_columns = required_columns - set(gwas_results.columns)

    if missing_columns:
        raise ValueError(
            f"Faltan columnas en gwas_results: {missing_columns}"
        )

    df = gwas_results.copy()

    # ============================================================
    # 3. Orden y tamaño aproximado de cromosomas
    # ============================================================

    chromosomes = pd.unique(df["Chromosome"])

    chromosome_sizes = {
        chromosome: df.loc[
            df["Chromosome"] == chromosome,
            "Position",
        ].max()
        for chromosome in chromosomes
    }

    # ============================================================
    # 4. Offsets
    # ============================================================

    chromosome_offsets = {}

    current_offset = 0

    for chromosome in chromosomes:
        chromosome_offsets[chromosome] = current_offset
        current_offset += chromosome_sizes[chromosome]

    df["Manhattan_position"] = (
        df["Position"]
        + df["Chromosome"].map(chromosome_offsets)
    )

    # ============================================================
    # 5. Centros y finales de los cromosomas
    # ============================================================

    chromosome_centers = []
    chromosome_boundaries = []

    for chromosome in chromosomes:

        start = chromosome_offsets[chromosome]
        end = start + chromosome_sizes[chromosome]

        chromosome_centers.append(
            (start + end) / 2
        )

        chromosome_boundaries.append(end)

    # ============================================================
    # 6. Colores
    # ============================================================

    cmap = plt.get_cmap("rainbow")

    colors = cmap(
        np.linspace(0, 1, len(chromosomes))
    )

    # ============================================================
    # 7. Crear figura
    # ============================================================

    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    for chromosome, color in zip(
        chromosomes,
        colors,
    ):

        chromosome_df = df[
            df["Chromosome"] == chromosome
        ].sort_values("Position")

        ax.scatter(
            chromosome_df["Manhattan_position"],
            chromosome_df[y_column],
            s=8,
            alpha=0.8,
            color=color,
            rasterized=True,
        )

    # ============================================================
    # 8. Límites entre cromosomas
    # ============================================================

    # No dibujamos una línea después del último cromosoma
    for boundary in chromosome_boundaries[:-1]:
        ax.axvline(
            x=boundary,
            linewidth=0.6,
            linestyle="--",
            color="gray",
            alpha=0.30,
        )

    # ============================================================
    # 9. Umbral de significancia
    # ============================================================

    if show_significance_threshold:

        if y_axis_variable == "p":
            num_tests = len(gwas_results)
            threshold = -np.log10(alpha / num_tests)
            threshold_label = "Umbral de significancia Bonferroni"

            # Ponemos también una línea para el alfa sin corregir
            raw_threshold = -np.log10(alpha)
            ax.axhline(
                y=raw_threshold,
                linestyle="--",
                linewidth=2.5,
                color="gray",
                alpha=0.8,
                label=f"Umbral de significancia no corregido (alfa = {alpha:g})",
            )
            # Para que salga a la derecha el valor
            ax.text(
                1.005,
                raw_threshold,
                f"{raw_threshold:.2f}",
                transform=ax.get_yaxis_transform(),
                va="center",
                ha="left",
                color="gray",
                fontweight="bold",
            )

            # Ponemos también un umbral de Benjamini-Hochberg FDR

            # Será el mayor p-valor que sigue siendo significativo según BH-FDR para 
            # una alfa dada (es decir, menor -log10(p) entre los SNPs significativos)
            _fdr_significant = (
                gwas_results["-log10(FDR_BH)"] >= -np.log10(alpha)
                )

            # No hay un alfa fijo. 
            # Mostramos la línea solo si algún -log10(p) corregido FDR supera el alfa nominal
            if _fdr_significant.any():

                fdr_threshold = gwas_results.loc[
                    _fdr_significant,
                    "-log10(p)"
                ].min()

                ax.axhline(
                    y=fdr_threshold,
                    linestyle="--",
                    linewidth=2.5,
                    color="tab:orange",
                    alpha=0.8,
                    label=f"Umbral de significancia BH-FDR",
                )

                ax.text(
                    1.005,
                    fdr_threshold,
                    f"{fdr_threshold:.2f}",
                    transform=ax.get_yaxis_transform(),
                    va="center",
                    ha="left",
                    color="tab:orange",
                    fontweight="bold"
                )

        elif y_axis_variable == "bonferroni":
            threshold = -np.log10(alpha)
            threshold_label = f"Umbral de significancia (alfa = {alpha:g})"

        elif y_axis_variable == "fdr":
            threshold = -np.log10(alpha)
            threshold_label = f"Umbral de significancia (alfa = {alpha:g})"

        ax.axhline(
            y=threshold,
            linestyle="--",
            linewidth=2.5,
            color="black",
            alpha=0.8,
            label=threshold_label,
        )
        # Para que salga a la derecha el valor
        ax.text(
            1.005,
            threshold,
            f"{threshold:.2f}",
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="left",
            fontweight="bold",
        )

        ax.legend(loc="upper left", fontsize=8.5)
    # ============================================================
    # 10. Formato
    # ============================================================

    ax.set_xticks(chromosome_centers)
    ax.set_xticklabels(chromosomes)

    ax.set_xlabel("Chromosome")
    ax.set_ylabel(y_column)

    ax.set_title(
        f"Manhattan plot — {phenotype_name}"
    )

    ax.tick_params(
        axis="x",
        rotation=45,
    )

    ax.margins(x=0.01)

    fig.tight_layout()

    return fig, ax


def create_qq_plot(
    gwas_results: pd.DataFrame,
    phenotype_name: str,
):
    """
    Crea un Q-Q plot a partir del DataFrame generado por do_gwas.

    El gráfico compara los p-valores observados en el GWAS con los
    p-valores esperados bajo la hipótesis nula de ausencia de asociación.

    Parameters
    ----------
    gwas_results : pd.DataFrame
        DataFrame generado por do_gwas.

    phenotype_name : str
        Nombre del carácter/fenotipo analizado.

    Returns
    -------
    fig, ax
        Figura y eje de Matplotlib.
    """

    # ============================================================
    # 1. Comprobar columna necesaria
    # ============================================================

    if "-log10(p)" not in gwas_results.columns:
        raise ValueError(
            "gwas_results debe contener la columna '-log10(p)'."
        )

    # ============================================================
    # 2. Obtener valores observados válidos
    # ============================================================

    observed_logp = pd.to_numeric(
        gwas_results["-log10(p)"],
        errors="coerce",
    ).to_numpy()

    observed_logp = observed_logp[
        np.isfinite(observed_logp)
    ]

    if len(observed_logp) == 0:
        raise ValueError(
            "No hay p-valores válidos para generar el Q-Q plot."
        )

    # ============================================================
    # 3. Ordenar p-valores observados
    # ============================================================

    # Valores más significativos primero
    observed_logp = np.sort(observed_logp)

    num_tests = len(observed_logp)

    # ============================================================
    # Factor de inflación genómica lambda_GC
    # ============================================================

    p_values = 10 ** (-observed_logp)

    chi2_values = stats.chi2.isf(
        p_values,
        df=1,
    )

    lambda_gc = (
        np.median(chi2_values)
        / stats.chi2.ppf(0.5, df=1)
    )

    # median_observed_logp = np.median(observed_logp)

    # ============================================================
    # 4. P-valores esperados bajo H0
    # ============================================================

    expected_p = (
        np.arange(1, num_tests + 1)
        / (num_tests + 1)
    )

    expected_logp = -np.log10(expected_p)

    # Queremos ambos ejes en orden creciente
    expected_logp = np.sort(expected_logp)

    # ============================================================
    # 5. Crear figura
    # ============================================================

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    ax.scatter(
        expected_logp,
        observed_logp,
        s=10,
        alpha=0.6,
        rasterized=True,
    )

    # ============================================================
    # 6. Recta esperada y = x
    # ============================================================

    max_expected = expected_logp.max()

    ax.plot(
        [0, max_expected],
        [0, max_expected],
        "k--",
        linewidth=1,
        label="Expected under H₀",
    )

    # ============================================================
    # 7. Formato
    # ============================================================

    ax.set_xlabel(
        "Expected -log10(p)"
    )

    ax.set_ylabel(
        "Observed -log10(p)"
    )

    ax.set_title(
        f"Q-Q plot — {phenotype_name}"
    )

    # Mostrar el factor de inflación lambda
    ax.text(
        0.05,
        0.90,
        (
            rf"$\lambda_{{GC}}$ = {lambda_gc:.3f}"
            # f"\nMedian of the observed -log10(p) = {median_observed_logp:.3f}"
        ),
        transform=ax.transAxes,
        va="top",
    )

    ax.legend()

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()

    return fig, ax

# Función PC plot(pca_df)
def create_pca_plot(
    covariates: pd.DataFrame,
):
    """
    Representa los dos primeros componentes principales de los individuos.

    Parameters
    ----------
    covariates : pd.DataFrame
        DataFrame de covariables obtenido del PCA.
        Las filas corresponden a individuos y las columnas a PCs.

    Returns
    -------
    fig, ax
        Figura y eje de Matplotlib.
    """

    if covariates.shape[1] < 2:
        raise ValueError("Se necesitan al menos dos componentes principales.")

    pc1 = covariates.iloc[:, 0]
    pc2 = covariates.iloc[:, 1]

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(
        pc1,
        pc2,
        s=30,
        alpha=0.7,
    )

    ax.axhline(
        y=0,
        color="gray",
        linewidth=0.8,
        alpha=0.5,
    )

    ax.axvline(
        x=0,
        color="gray",
        linewidth=0.8,
        alpha=0.5,
    )

    max_abs_x = max(abs(pc1.min()), abs(pc1.max())) * 1.05
    max_abs_y = max(abs(pc2.min()), abs(pc2.max())) * 1.05

    ax.set_xlim(-max_abs_x, max_abs_x)
    ax.set_ylim(-max_abs_y, max_abs_y)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    ax.set_title(
        "PCA — PC1 vs PC2"
    )

    fig.tight_layout()

    return fig, ax