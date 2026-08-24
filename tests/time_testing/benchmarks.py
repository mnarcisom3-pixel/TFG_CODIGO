import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pynei

import gwaslib.quantitative as gw_quant
import gwaslib.qualitative as gw_qual


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

# CAMBIAR ÚNICAMENTE ESTE PARÁMETRO:
#
# "benchmark_linreg"
# "benchmark_logreg"

output_file_name = "benchmark_linreg"


if output_file_name not in {
    "benchmark_linreg",
    "benchmark_logreg",
}:
    raise ValueError(
        "output_file_name debe ser "
        "'benchmark_linreg' o 'benchmark_logreg'."
    )


np.random.seed(12345)

M_values = [
    100,
    500,
    1000,
    2500,
    5000,
    10000,
    25000,
]

K_pcs = 10
n_runs = 10


# ============================================================
# 2. CONFIGURACIÓN SEGÚN TIPO DE GWAS
# ============================================================

if output_file_name == "benchmark_linreg":

    analysis_type = "quantitative"

    N_indivs = 200

    methods = {
        "sm": gw_quant.linreg_sm,
        "nploop": gw_quant.linreg_nploop,
        "3d": gw_quant.linreg_3d,
    }

    table_names = {
        "sm": "linreg_sm (ms)",
        "nploop": "linreg_nploop (ms)",
        "3d": "linreg_3d (ms)",
    }

    plot_names = {
        "sm": "Statsmodels loop",
        "nploop": "NumPy loop",
        "3d": "NumPy 3D",
    }

    plot_title = (
        "Tiempo de ejecución del GWAS cuantitativo sobre un dataset simulado"
    )

else:

    analysis_type = "qualitative"

    N_indivs = 200

    methods = {
        "sm": gw_qual.logreg_sm,
        "nploop": gw_qual.logreg_nploop,
        "3d": gw_qual.logreg_3d,
    }

    table_names = {
        "sm": "logreg_sm (ms)",
        "nploop": "logreg_nploop (ms)",
        "3d": "logreg_3d (ms)",
    }

    plot_names = {
        "sm": "Statsmodels loop",
        "nploop": "NumPy loop",
        "3d": "NumPy 3D",
    }

    plot_title = (
        "Tiempos de ejecución del GWAS cualitativo sobre un dataset simulado"
    )


# ============================================================
# 3. GENERACIÓN DEL DATASET SIMULADO
# ============================================================

M_max = max(M_values)

print(
    f"\nGenerando dataset {analysis_type}: "
    f"{M_max} SNPs, "
    f"{N_indivs} individuos, "
    f"{K_pcs} PCs"
)


# ------------------------------------------------------------
# Genotipos realistas según Hardy-Weinberg
# ------------------------------------------------------------

p_alt = np.random.uniform(
    0.05,
    0.5,
    size=(M_max, 1),
)

prob_0 = (1 - p_alt) ** 2
prob_1 = 2 * p_alt * (1 - p_alt)
prob_2 = p_alt ** 2

probs = np.hstack([
    prob_0,
    prob_1,
    prob_2,
])


matriz012_max = np.empty(
    (M_max, N_indivs)
)

for idx_snp in range(M_max):

    matriz012_max[idx_snp, :] = np.random.choice(
        [0, 1, 2],
        size=N_indivs,
        p=probs[idx_snp],
    )



# ============================================================
# 5. GENERACIÓN DEL FENOTIPO
# ============================================================

if analysis_type == "quantitative":

    # --------------------------------------------------------
    # Fenotipo cuantitativo
    # --------------------------------------------------------

    phenotypes_test = (
        170.0
        + 2.0 * matriz012_max[0, :]
        - 1.5 * matriz012_max[1, :]
        + 2.5 * matriz012_max[2, :]
        - 4.0 * matriz012_max[3, :]
        + 0.7 * matriz012_max[4, :]
        + np.random.normal(
            loc=0.0,
            scale=4.0,
            size=N_indivs,
        )
    )

    phenotypes_test = np.clip(
        phenotypes_test,
        140.0,
        210.0,
    )


else:

    # --------------------------------------------------------
    # Fenotipo cualitativo
    #
    # 0 = control
    # 1 = caso
    # --------------------------------------------------------

    # IMPORTANTE
    # Se escogen (al azar) SNPs que no producen separación en este dataset simulado,
    # para poder comparar el rendimiento de las tres implementaciones
    # bajo condiciones en las que todas convergen correctamente.
    eta_real = (
        -2.8
        + 1.6 * matriz012_max[0, :]
        - 1.4 * matriz012_max[1, :]
        + 1.5 * matriz012_max[98, :]
        - 1.5 * matriz012_max[3, :]
        + 1.3 * matriz012_max[4, :]
    )

    prob_real = (
        1.0
        / (1.0 + np.exp(-eta_real))
    )

    phenotypes_test = np.random.binomial(
        n=1,
        p=prob_real,
    )

    print(
        "Controles:",
        np.sum(phenotypes_test == 0),
    )

    print(
        "Casos:",
        np.sum(phenotypes_test == 1),
    )


# ============================================================
# 6. BENCHMARK
# ============================================================

summary_rows = []
raw_rows = []


# Aleatorizamos el orden de ejecución para que ninguna función
# se beneficie sistemáticamente de ejecutarse antes o después.
order_rng = np.random.default_rng(98765)


for M_snps in M_values:

    print(
        "\n"
        "============================================\n"
        f"Benchmark: M = {M_snps} SNPs\n"
        "============================================"
    )

    matriz012_test = matriz012_max[
        :M_snps,
        :
    ]

    # PCA específico para este tamaño M
    mat012_for_pca = pd.DataFrame(
        matriz012_test.T
    )

    PCA = pynei.pca.do_pca(
        mat012_for_pca
    )

    covariables_test = (
        PCA["projections"]
        .to_numpy()[:, :K_pcs]
    )

    # ========================================================
    # 6.1. WARM-UP + COMPROBACIÓN DE EQUIVALENCIA
    # ========================================================

    print(
        "Comprobando equivalencia numérica..."
    )

    warm_results = {}

    for name, function in methods.items():

        warm_results[name] = function(
            matriz012_test,
            phenotypes_test,
            covariables_test,
        )


    # Statsmodels vs NumPy loop
    assert np.allclose(
        warm_results["sm"]["beta"],
        warm_results["nploop"]["beta"],
        rtol=1e-5,
        atol=1e-12,
    )
    
    assert np.allclose(
        warm_results["sm"]["SE"],
        warm_results["nploop"]["SE"],
        rtol=1e-5,
        atol=1e-12,
    )
    
    assert np.allclose(
        warm_results["sm"]["p_val"],
        warm_results["nploop"]["p_val"],
        rtol=1e-5,
        atol=1e-12,
    )

    assert np.allclose(
        warm_results["nploop"]["beta"],
        warm_results["3d"]["beta"],
        rtol=1e-5,
        atol=1e-12,
    )

    assert np.allclose(
        warm_results["nploop"]["SE"],
        warm_results["3d"]["SE"],
        rtol=1e-5,
        atol=1e-12,
    )

    assert np.allclose(
        warm_results["nploop"]["p_val"],
        warm_results["3d"]["p_val"],
        rtol=1e-5,
        atol=1e-12,
    )

    print(
        "Resultados numéricos equivalentes: OK"
    )

    # Si hubiera problemas con el np.allclose
    '''
    a = warm_results["sm"]["SE"]
    b = warm_results["nploop"]["SE"]

    print("NaN sm:", np.isnan(a).sum())
    print("NaN nploop:", np.isnan(b).sum())

    diff = np.abs(a - b)

    print("Máxima diferencia absoluta:", np.nanmax(diff))
    print(
        "Índice máxima diferencia:",
        np.nanargmax(diff),
    )

    idx = np.nanargmax(diff)

    print("SE sm:", a[idx])
    print("SE nploop:", b[idx])

    print(
        "allclose normal:",
        np.allclose(
            a,
            b,
            rtol=1e-5,
            atol=1e-12,
        )
    )

    print(
        "allclose ignorando NaN coincidentes:",
        np.allclose(
            a,
            b,
            rtol=1e-5,
            atol=1e-12,
            equal_nan=True,
        )
    )
    '''

    # ========================================================
    # 6.2. MEDICIÓN DE TIEMPOS
    # ========================================================

    times = {
        name: []
        for name in methods
    }


    for run in range(n_runs):

        print(
            f"Repetición {run + 1}/{n_runs}"
        )

        method_order = order_rng.permutation(
            list(methods.keys())
        )


        for name in method_order:

            function = methods[name]

            t0 = time.perf_counter()

            function(
                matriz012_test,
                phenotypes_test,
                covariables_test,
            )

            elapsed_ms = (
                time.perf_counter() - t0
            ) * 1000


            times[name].append(
                elapsed_ms
            )


            raw_rows.append({
                "M_snps": M_snps,
                "run": run + 1,
                "method": name,
                "time_ms": elapsed_ms,
            })


    # ========================================================
    # 6.3. MEDIA Y DESVIACIÓN ESTÁNDAR
    # ========================================================

    mean_sm = np.mean(
        times["sm"]
    )

    sd_sm = np.std(
        times["sm"],
        ddof=1,
    )


    mean_loop = np.mean(
        times["nploop"]
    )

    sd_loop = np.std(
        times["nploop"],
        ddof=1,
    )


    mean_3d = np.mean(
        times["3d"]
    )

    sd_3d = np.std(
        times["3d"],
        ddof=1,
    )


    # ========================================================
    # 6.4. SPEED-UP
    # ========================================================

    # > 1  -> 3D es más rápida
    # = 1  -> rendimiento equivalente
    # < 1  -> nploop es más rápida

    speedup_3d_vs_sm = (
        mean_sm / mean_3d
    )

    speedup_3d_vs_loop = (
        mean_loop / mean_3d
    )

    summary_rows.append({

        "M_snps": M_snps,

        "sm_mean": mean_sm,
        "sm_sd": sd_sm,

        "nploop_mean": mean_loop,
        "nploop_sd": sd_loop,

        "3d_mean": mean_3d,
        "3d_sd": sd_3d,

        "speedup_3d_vs_sm":
            speedup_3d_vs_sm,

        "speedup_3d_vs_nploop":
            speedup_3d_vs_loop,
    })


# ============================================================
# 7. DATAFRAMES DE RESULTADOS
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

raw_df = pd.DataFrame(
    raw_rows
)

summary_df.to_csv(
    f"{output_file_name}_summary.csv",
    index=False,
)

raw_df.to_csv(
    f"{output_file_name}_raw.csv",
    index=False,
)


# ============================================================
# 8. TABLA PARA LA MEMORIA
# ============================================================

table_df = pd.DataFrame({

    "M (SNPs)":
        summary_df["M_snps"],

    table_names["sm"]: [
        f"{mean:.2f} ± {sd:.2f}"
        for mean, sd in zip(
            summary_df["sm_mean"],
            summary_df["sm_sd"],
        )
    ],

    table_names["nploop"]: [
        f"{mean:.2f} ± {sd:.2f}"
        for mean, sd in zip(
            summary_df["nploop_mean"],
            summary_df["nploop_sd"],
        )
    ],

    table_names["3d"]: [
        f"{mean:.2f} ± {sd:.2f}"
        for mean, sd in zip(
            summary_df["3d_mean"],
            summary_df["3d_sd"],
        )
    ],

    "Speed-up 3D vs sm-loop": [
        f"{value:.2f}×"
        for value in summary_df[
            "speedup_3d_vs_sm"
        ]
    ],

    "Speed-up 3D vs np-loop": [
        f"{value:.2f}×"
        for value in summary_df[
            "speedup_3d_vs_nploop"
        ]
    ],
})


print("\nResultados del benchmark:\n")

print(
    table_df.to_string(
        index=False
    )
)


# ============================================================
# 9. TABLA COMO PNG Y PDF
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 4.5)
)

ax.axis("off")


table = ax.table(

    cellText=table_df.values,

    colLabels=table_df.columns,

    cellLoc="center",

    colLoc="center",

    loc="center",
)


table.auto_set_font_size(False)

table.set_fontsize(10)

table.scale(1, 1.6)


# ------------------------------------------------------------
# Encabezados en negrita
# ------------------------------------------------------------

for column_idx in range(
    len(table_df.columns)
):

    table[
        (0, column_idx)
    ].set_text_props(
        weight="bold"
    )

ax.set_title(
    f"{plot_title} "
    f"(media ± SD, n = {n_runs})",
    pad=15,
    fontweight='bold',
)

fig.tight_layout()

fig.savefig(
    f"{output_file_name}_table.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    f"{output_file_name}_table.pdf",
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 10. GRÁFICA DE RENDIMIENTO
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 6)
)


for name in methods:

    ax.errorbar(
        summary_df["M_snps"],
        summary_df[f"{name}_mean"],
        yerr=summary_df[f"{name}_sd"],
        marker="o",
        capsize=4,
        linewidth=1.5,
        label=plot_names[name],
    )


# ============================================================
# 11. FORMATO DE LA GRÁFICA
# ============================================================

ax.set_xscale("log")

ax.set_yscale("log")

ax.set_xticks(M_values)

ax.set_xticklabels(
    [
        str(M)
        for M in M_values
    ]
)


ax.set_xlabel("Número de SNPs")

ax.set_ylabel("Tiempo de ejecución (ms)")

ax.set_title(plot_title, fontweight='bold')

ax.legend(title="Función utilizada").get_title().set_weight('bold')

ax.grid(alpha=0.25, which="both")

fig.tight_layout()


# ============================================================
# 12. GUARDAR GRÁFICA
# ============================================================

fig.savefig(
    f"{output_file_name}_performance.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    f"{output_file_name}_performance.pdf",
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 13. RESUMEN FINAL
# ============================================================

print("\nBenchmark terminado.")

print("\nArchivos generados:")
print(f"  {output_file_name}_summary.csv")
print(f"  {output_file_name}_raw.csv")
print(f"  {output_file_name}_table.png")
print(f"  {output_file_name}_table.pdf")
print(f"  {output_file_name}_performance.png")
print(f"  {output_file_name}_performance.pdf")