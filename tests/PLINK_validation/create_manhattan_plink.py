from gwaslib import visualization

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.stats.multitest import multipletests

input_path = "Varitome_gwas_plink2_quanti2_ownpcs.PHENO.glm.linear"
output_png = "Manhattan_PLINK_mean_color_b.png"
phenotype_name = "PLINK2 \n(Mean Color b)"
'''

input_path = "Varitome_gwas_plink2_quali_ownpcs.PHENO.glm.logistic.hybrid"
output_png = "Manhattan_PLINK_inflorescence_forked.png"
phenotype_name = "PLINK2 \n(Inflorescence Forked Type)"
'''
'''
uv run python create_manhattan_plink.py `   
    Varitome_gwas_plink2_quanti_ownpcs.PHENO.glm.linear `                                
    Manhattan_PLINK_fruit_weight.png `                                                                             
    "PLINK2 Fruit Weight"
'''

# ============================================================
# 1. Leer resultados de PLINK
# ============================================================

df = pd.read_csv(input_path, sep="\t")

# Solo efecto aditivo del SNP
df = df[df["TEST"] == "ADD"].copy()

# Seleccionamos solo los SNPs con p-valores válidos (finitos) de PLINK
# Vamos a eliminar los SNPs que tenían p-valores inválidos en PLINK,
# (eliminarlos del dataframe que devuelve do_gwas)
my_gwas_results = pd.read_csv("Varitome_own_results_quanti_2.csv")

plink_p = pd.to_numeric(
    df["P"],
    errors="coerce",
).to_numpy()

valid_common = (
    np.isfinite(plink_p)
    & (plink_p > 0)
    & (plink_p <= 1)
)

df = df.loc[valid_common].copy() # este es el df de resultados de plink, "limpio" de los SNPs inválidos
gwaslib_common_snps = my_gwas_results.loc[valid_common].copy()

# Creamos directamente el Manhattan plot de gwaslib, con el dataframe "limpio"

fig, ax = visualization.create_manhattan_plot(
    gwaslib_common_snps,
    y_axis_variable="p",
    phenotype_name="GWASLIB \n(Mean Color b)"
)

fig.savefig(
    "Manhattan_GWASLIB_mean_color_b.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    "Manhattan_GWASLIB_mean_color_b.pdf",
    bbox_inches="tight",
)

plt.close(fig)


# Ahora cosntruimos el Manhattan plot con el df "limpio" de PLINK
# Cromosoma y posición
df["#CHROM"] = pd.to_numeric(df["#CHROM"])
df["POS"] = pd.to_numeric(df["POS"])


# ============================================================
# 2. Orden y tamaño aproximado de cromosomas
# ============================================================

chromosomes = pd.unique(df["#CHROM"])

chromosome_sizes = {
    chromosome: df.loc[
        df["#CHROM"] == chromosome,
        "POS",
    ].max()
    for chromosome in chromosomes
}


# ============================================================
# 3. Offsets
# ============================================================

chromosome_offsets = {}

current_offset = 0

for chromosome in chromosomes:
    chromosome_offsets[chromosome] = current_offset
    current_offset += chromosome_sizes[chromosome]

df["Manhattan_position"] = (
    df["POS"]
    + df["#CHROM"].map(chromosome_offsets)
)


# ============================================================
# 4. Centros y límites de cromosomas
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
# 5. Variable eje Y
# ============================================================

df["-log10(p)"] = -np.log10(df["P"])

# Esta no saldrá en el eje Y, pero la sacamos para el umbral
df["FDR_BH"] = multipletests(df["P"].to_numpy(), method="fdr_bh")[1]

# ============================================================
# 6. Colores
# ============================================================

cmap = plt.get_cmap("rainbow")

colors = cmap(
    np.linspace(0, 1, len(chromosomes))
)


# ============================================================
# 7. Figura
# ============================================================

fig, ax = plt.subplots(
    figsize=(14, 6)
)

for chromosome, color in zip(
    chromosomes,
    colors,
):

    chromosome_df = df[
        df["#CHROM"] == chromosome
    ].sort_values("POS")

    ax.scatter(
        chromosome_df["Manhattan_position"],
        chromosome_df["-log10(p)"],
        s=8,
        alpha=0.8,
        color=color,
        rasterized=True,
    )


# ============================================================
# 8. Límites entre cromosomas
# ============================================================

for boundary in chromosome_boundaries[:-1]:

    ax.axvline(
        x=boundary,
        linewidth=0.6,
        linestyle="--",
        color="gray",
        alpha=0.30,
    )


# ============================================================
# 9. Umbral genome-wide Bonferroni
# ============================================================

alpha = 0.05



# Para mostrar una línea también para alfa sin corregir
raw_threshold = -np.log10(alpha)
ax.axhline(
    y=raw_threshold,
    linestyle="--",
    linewidth=2.5,
    color="gray",
    alpha=0.8,
    label=f"Umbral de significancia no corregido (alfa = {alpha:g})",
)

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

# Para mostrar el umbral correspondiente a Benjamini-Hochberg FDR
fdr_significant = df["FDR_BH"] <= alpha

if fdr_significant.any():

    # Mayor p-valor crudo que sigue siendo significativo según BH-FDR
    # = menor -log10(p) entre los SNPs significativos
    fdr_threshold = df.loc[
        fdr_significant,
        "-log10(p)"
    ].min()

    ax.axhline(
        y=fdr_threshold,
        linestyle="--",
        linewidth=2.5,
        color="tab:orange",
        alpha=0.8,
        label="Umbral de significancia BH-FDR",
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

# Para mostrar una línea para umbral Bonferroni
num_tests = len(df)

threshold = -np.log10(
    alpha / num_tests
)

ax.axhline(
    y=threshold,
    linestyle="--",
    linewidth=2.5,
    color="black",
    alpha=0.8,
    label="Umbral de significancia Bonferroni",
)

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
ax.set_xticklabels(
    [f"SL2.50ch{int(chrom):02d}" for chrom in chromosomes]
)

ax.set_xlabel("Chromosome")
ax.set_ylabel("-log10(p)")

ax.set_title(
    f"Manhattan plot — {phenotype_name}"
)

ax.tick_params(
    axis="x",
    rotation=45,
)

ax.margins(x=0.01)

fig.tight_layout()


# ============================================================
# 11. Guardar PNG y PDF
# ============================================================

fig.savefig(
    output_png,
    dpi=300,
    bbox_inches="tight",
)

output_pdf = output_png.rsplit(".", 1)[0] + ".pdf"

fig.savefig(
    output_pdf,
    bbox_inches="tight",
)

plt.close(fig)

print(f"Creado: {output_png}")
print(f"Creado: {output_pdf}")


'''
uv run python create_manhattan_plink.py `   
>>     Varitome_gwas_plink2_quali_ownpcs.PHENO.glm.logistic.hybrid `                        
>>     Manhattan_PLINK_inflorescence_forked.png `                                                                     
>>     "PLINK2 Inflorescence Forked Type"
'''