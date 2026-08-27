import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
async def _():
    import sys


    if sys.platform == "emscripten":

        import micropip

        await micropip.install([
            "scipy",
            "statsmodels",
            "more-itertools",
            "matplotlib",
            "python-calamine",
        ])

    import gwaslib as gw
    import pynei

    return gw, pynei


@app.cell
def _(gw, mo, pynei):
    mo.md(f"""
    # GWASLIB Web

    ✅ Imports successful

    - gwaslib: `{gw.__file__}`
    - pynei: `{pynei.__file__}`
    """)
    return


@app.cell
def _(mo):
    import numpy as np


    x = np.array([1, 2, 3])


    mo.md(f"NumPy OK: `{x.sum()}`")
    return (np,)


@app.cell
def _(mo, np, pynei):
    import pandas as pd

    np.random.seed(12345)

    mat_012 = np.random.randint(
        0,
        3,
        size=(100, 200),
    )

    PCA = pynei.pca.do_pca(
        pd.DataFrame(mat_012.T)
    )

    mo.md(
        f"Pynei PCA OK: `{PCA['projections'].shape}`"
    )
    return PCA, pd


@app.cell
def _(gw, mo, np):
    np.random.seed(12345)

    M = 100
    N = 200

    matriz012 = np.random.randint(
        0,
        3,
        size=(M, N),
    )

    phenotypes = np.random.normal(
        size=N,
    )

    covariates = np.random.normal(
        size=(N, 10),
    )

    result = gw.quantitative.linreg_3d(
        matriz012,
        phenotypes,
        covariates,
    )

    mo.md(
        f"gwaslib linreg_3d OK: `{len(result['p_val'])}` SNPs"
    )
    return


@app.cell
def _(PCA, gw, mo, np, pd, pynei):
    import matplotlib.pyplot as plt
    import time

    # =====================================================================
    # 2. GENERACIÓN DE DATOS SIMULADOS REALISTAS (20 SNPs, 200 Indivs, 10 PCs)
    # =====================================================================
    np.random.seed(12345)

    M_snps = 15000
    N_indivs = 200
    K_pcs = 10

    print(
        f"--- 1. Generando datos coherentes: {M_snps} SNPs, {N_indivs} individuos,"
        f" {K_pcs} PCs ---"
    )

    # A. Genotipos realistas según Hardy-Weinberg (Frecuencias p entre 0.1 y 0.5)
    p_alt = np.random.uniform(0.05, 0.5, size=(M_snps, 1))
    prob_0 = (1 - p_alt) ** 2
    prob_1 = 2 * p_alt * (1 - p_alt)
    prob_2 = p_alt**2
    probs = np.hstack([prob_0, prob_1, prob_2])

    matriz012_test = np.empty((M_snps, N_indivs))
    for idx_snp in range(M_snps):
      matriz012_test[idx_snp, :] = np.random.choice(
          [0, 1, 2], size=N_indivs, p=probs[idx_snp]
      )

    # Covariables: PCA con pynei sobre nuestra matriz012
    mat012_forPCA = pd.DataFrame(matriz012_test.T)
    PCs = pynei.pca.do_pca(mat012_forPCA)

    covariables_test = PCs["projections"].to_numpy()[:, :10]
    ''''''
    print(covariables_test.shape)
    plt.scatter(covariables_test[:,0],covariables_test[:,1])
    plt.show()


    '''
    # B. Covariables: PC1 refleja estructura poblacional (ancestría), PC2-PC10 ruido biológico
    pc1 = np.random.normal(loc=0.0, scale=2.0, size=N_indivs)
    pcs_resto = np.random.normal(loc=0.0, scale=1.0, size=(N_indivs, K_pcs - 1))
    covariables_test = np.column_stack([pc1, pcs_resto])
    '''
    # C. Fenotipo de altura (cm): Base + Efecto Ancestría (PC1) + Ruido ambiental
    # D. INTRODUCIMOS EL ASOCIACIÓN REAL: Solo el SNP 0 tiene efecto (+4.0 cm por alelo)
    # Los SNPs del 1 al M no tienen ningún efecto sobre Y
    phenotypes_test = (
        170.0
    #    + (3.0 * covariables_test[:, 0])
        + 4.0 * matriz012_test[0, :]
        + np.random.normal(loc=0.0, scale=4.0, size=N_indivs)
    )
    phenotypes_test = np.clip(phenotypes_test, 140.0, 210.0)


    # =====================================================================
    # 3. EJECUCIÓN, TIEMPOS Y COMPROBACIÓN DE IDENTIDAD PARA EL TUTOR
    # =====================================================================

    print("--- 2. Ejecutando regresión con bucle y STATSMODELS... ---")
    t0 = time.perf_counter()
    res_sm = gw.quantitative.linreg_sm(matriz012_test, phenotypes_test, covariables_test)
    t_sm = (time.perf_counter() - t0) * 1000

    print("--- 3. Ejecutando regresión con bucle y NUMPY... ---")
    t0 = time.perf_counter()
    res_bucle = gw.quantitative.linreg_nploop(matriz012_test, phenotypes_test, covariables_test)
    t_bucle = (time.perf_counter() - t0) * 1000

    print("--- 4. Ejecutando regresión 3D vectorizada... ---")
    t0 = time.perf_counter()
    res_3d = gw.quantitative.linreg_3d(matriz012_test, phenotypes_test, covariables_test)
    t_3d = (time.perf_counter() - t0) * 1000


    mo.vstack([t_sm, t_bucle, t_3d])
    return


if __name__ == "__main__":
    app.run()
