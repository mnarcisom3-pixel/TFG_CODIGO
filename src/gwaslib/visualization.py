import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_manhattan_plot(
    gwas_results: pd.DataFrame,
    phenotype_name: str,
    y_axis_variable: str = "p",
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
        a un nivel de significancia alpha = 0.05.

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

        alpha = 0.05

        if y_axis_variable == "p":
            num_tests = len(gwas_results)
            threshold = -np.log10(alpha / num_tests)
            threshold_label = "Genome-wide significance threshold (Bonferroni-corrected)"

        elif y_axis_variable == "bonferroni":
            threshold = -np.log10(alpha)
            threshold_label = "Bonferroni-adjusted p = 0.05"

        elif y_axis_variable == "fdr":
            threshold = -np.log10(alpha)
            threshold_label = "BH-FDR adjusted p = 0.05"

        ax.axhline(
            y=threshold,
            linestyle="--",
            linewidth=2.5,
            color="black",
            alpha=0.7,
            label=threshold_label,
        )

        ax.legend()
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

# do_manhattan_plot()

# Función QQ plot


# Función PC plot(pca_dict)