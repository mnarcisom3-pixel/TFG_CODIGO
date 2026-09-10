from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# INPUT
# ============================================================

input_csvs = [
    Path("timing_20_MB.csv"),
    Path("timing_50_MB.csv"),
    Path("timing_100_MB.csv"),
]


# ============================================================
# LEER RESULTADOS
# ============================================================

results = pd.concat(
    [pd.read_csv(path) for path in input_csvs],
    ignore_index=True,
)


# ============================================================
# TABLA PARA LA MEMORIA
# ============================================================

def mean_sd_minutes(row, stage):
    mean = row[f"{stage}_mean_s"] / 60
    sd = row[f"{stage}_sd_s"] / 60

    return f"{mean:.2f} ± {sd:.2f}"


table_df = pd.DataFrame({
    "VCF": results["VCF"],
    "Carga\n(min)": [
        mean_sd_minutes(row, "load")
        for _, row in results.iterrows()
    ],
    "Filtrado PCA\n(min)": [
        mean_sd_minutes(row, "pca_filter")
        for _, row in results.iterrows()
    ],
    "PCA\n(min)": [
        mean_sd_minutes(row, "pca")
        for _, row in results.iterrows()
    ],
    "Filtrado GWAS\n(min)": [
        mean_sd_minutes(row, "gwas_filter")
        for _, row in results.iterrows()
    ],
    "GWAS + gráficas\n(min)": [
        mean_sd_minutes(row, "gwas_graphs")
        for _, row in results.iterrows()
    ],
    "Tiempo total\n(min)": [
        mean_sd_minutes(row, "total")
        for _, row in results.iterrows()
    ],
})


# Guardamos también la tabla como CSV
table_df.to_csv(
    "gwas_timing_table.csv",
    index=False,
)


# Figura de la tabla
fig, ax = plt.subplots(figsize=(14, 3))
ax.axis("off")

table = ax.table(
    cellText=table_df.values,
    colLabels=table_df.columns,
    cellLoc="center",
    loc="center",
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.7)

# Cabecera en negrita
for column in range(len(table_df.columns)):
    table[(0, column)].set_text_props(weight="bold")

ax.set_title(
    "Perfil temporal del flujo de análisis GWAS",
    fontsize=12,
    fontweight="bold",
    pad=15,
)

fig.tight_layout()

fig.savefig(
    "gwas_timing_table.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    "gwas_timing_table.pdf",
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# BARRA APILADA AL 100 %
# ============================================================

stages = [
    ("load", "Carga de ficheros"),
    ("pca_filter", "Filtrado para PCA"),
    ("pca", "PCA"),
    ("gwas_filter", "Filtrado para GWAS"),
    ("gwas_graphs", "GWAS + gráficas"),
]
fig, ax = plt.subplots(figsize=(10, 4))

left = np.zeros(len(results))

for stage, label in stages:

    percentages = (
        results[f"{stage}_mean_s"]
        / results["total_mean_s"]
        * 100
    )

    ax.barh(
        results["VCF"],
        percentages,
        left=left,
        label=label,
    )

    left += percentages.to_numpy()


ax.set_xlabel("Porcentaje del tiempo total (%)")
ax.set_ylabel("Tamaño del VCF")

ax.set_xlim(0, 100)

ax.set_title(
    "Distribución del tiempo de ejecución por etapas"
)

ax.grid(
    axis="x",
    linestyle="--",
    alpha=0.3,
)

ax.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    frameon=False,
)

fig.tight_layout()

fig.savefig(
    "gwas_timing_stacked.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    "gwas_timing_stacked.pdf",
    bbox_inches="tight",
)

plt.close(fig)


print("Tabla y gráfica creadas correctamente.")