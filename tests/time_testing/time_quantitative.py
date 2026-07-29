import numpy as np
import time

import gwaslib.quantitative as gw_quant


# =====================================================================
# 2. GENERACIÓN DE DATOS SIMULADOS REALISTAS (20 SNPs, 200 Indivs, 10 PCs)
# =====================================================================
np.random.seed(12345)

M_snps = 10000
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
PCA = pynei.pca.do_pca(mat012_forPCA)

covariables_test = PCA["projections"].to_numpy()[:, :10]
# Si queremos podemos hacer un plot de PC1 vs PC2

'''
print(covariables_test.shape)
plt.scatter(covariables_test[:,0],covariables_test[:,1])
plt.show()
'''

'''
# B. Covariables: PC1 refleja estructura poblacional (ancestría), PC2-PC10 ruido biológico
pc1 = np.random.normal(loc=0.0, scale=2.0, size=N_indivs)
pcs_resto = np.random.normal(loc=0.0, scale=1.0, size=(N_indivs, K_pcs - 1))
covariables_test = np.column_stack([pc1, pcs_resto])
'''

# C. Fenotipo de altura (cm): Base + Efecto Ancestría (PC1) + Ruido ambiental
# INTRODUCIMOS 5 SNPS ASOCIADOS: Índices 0, 1, 2, 3 y 4 con distintos efectos (+ y -)
phenotypes_test = (
    170.0
    + (3.0 * covariables_test[:, 0])
    + (2.0 * matriz012_test[0, :])
    - (1.5 * matriz012_test[1, :])
    + (2.5 * matriz012_test[2, :])
    - (4.0 * matriz012_test[3, :])
    + (0.7 * matriz012_test[4, :])
    + np.random.normal(loc=0.0, scale=4.0, size=N_indivs)
)

# D. INTRODUCIMOS EL ASOCIACIÓN REAL: SNP 0 hasta el 4 tienen efecto (+4.0 cm por alelo)
# Los SNPs del 5 al 9999 no tienen ningún efecto sobre Y
phenotypes_test = np.clip(phenotypes_test, 140.0, 210.0)


# =====================================================================
# 3. EJECUCIÓN, TIEMPOS Y COMPROBACIÓN DE IDENTIDAD PARA EL TUTOR
# =====================================================================

n_runs = 5
print(
    f"\n--- 2. Ejecutando comparativa de rendimiento ({n_runs} repeticiones"
    f" para {M_snps} SNPs)... ---"
)
t_sm_list, t_bucle_list, t_3d_list = [], [], []

for r in range(n_runs):
    print(f"   Corriendo iteración {r + 1}/{n_runs}...")

    t0 = time.perf_counter()
    res_sm = gw_quant.linreg_sm(matriz012_test, phenotypes_test, covariables_test)
    t_sm_list.append((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    res_bucle = gw_quant.linreg_nploop(matriz012_test, phenotypes_test, covariables_test)
    t_bucle_list.append((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    res_3d = gw_quant.linreg_3d(matriz012_test, phenotypes_test, covariables_test)
    t_3d_list.append((time.perf_counter() - t0) * 1000)

# Cálculo de promedios y desviaciones estándar (ddof=1 para muestra muestral)
mean_sm, sd_sm = np.mean(t_sm_list), np.std(t_sm_list, ddof=1)
mean_bucle, sd_bucle = np.mean(t_bucle_list), np.std(t_bucle_list, ddof=1)
mean_3d, sd_3d = np.mean(t_3d_list), np.std(t_3d_list, ddof=1)

print("\n====================================================================")
print(
    f" COMPARATIVA DE TIEMPOS DE EJECUCIÓN (Promedio ± SD, {n_runs} corridas,"
    f" {M_snps} SNPs)"
)
print("====================================================================")
print(f" -> Tiempo Statsmodels (bucle): {mean_sm:9.2f} ± {sd_sm:6.2f} ms")
print(f" -> Tiempo NumPy (bucle):       {mean_bucle:9.2f} ± {sd_bucle:6.2f} ms")
print(f" -> Tiempo NumPy (3D):          {mean_3d:9.2f} ± {sd_3d:6.2f} ms")

acc_bucle_sm = mean_sm / mean_bucle
acc_3d_bucle = mean_bucle / mean_3d
acc_3d_sm = mean_sm / mean_3d

print(
    " [*] Aceleración Numpy (bucle) vs Statsmodels:"
    f" {acc_bucle_sm:.1f}x más rápido"
)
print(
    " [*] Aceleración Numpy 3D vs Numpy (bucle):   "
    f" {acc_3d_bucle:.1f}x más rápido"
)
print(
    " [*] Aceleración Numpy 3D vs Statsmodels:     "
    f" {acc_3d_sm:.1f}x más rápido"
)

print("\n====================================================================")
print(" VERIFICACIÓN DE IDENTIDAD MATEMÁTICA (Tolerancia np.allclose)")
print("====================================================================")
# Comprobamos que el método bucle da exactamente lo mismo que Statsmodels
betas_ok_1 = np.allclose(res_sm["beta"], res_bucle["beta"])
ses_ok_1 = np.allclose(res_sm["SE"], res_bucle["SE"])
pvals_ok_1 = np.allclose(res_sm["p_val"], res_bucle["p_val"])
# Comprobamos que el método 3D da exactamente lo mismo que el bucle
betas_ok_2 = np.allclose(res_bucle["beta"], res_3d["beta"])
ses_ok_2 = np.allclose(res_bucle["SE"], res_3d["SE"])
pvals_ok_2 = np.allclose(res_bucle["p_val"], res_3d["p_val"])

print("Comparación entre linreg_sm y linreg_nploop")
print(f" [*] ¿Coinciden las Betas al 100%?            -> {'SÍ' if betas_ok_1 else 'NO'}")
print(f" [*] ¿Coinciden los Errores Estándar al 100%? -> {'SÍ' if ses_ok_1 else 'NO'}")
print(f" [*] ¿Coinciden los P-valores al 100%?        -> {'SÍ' if pvals_ok_1 else 'NO'}")

print("\nComparación entre linreg_nploop y linreg_3d")
print(f" [*] ¿Coinciden las Betas al 100%?            -> {'SÍ' if betas_ok_2 else 'NO'}")
print(f" [*] ¿Coinciden los Errores Estándar al 100%? -> {'SÍ' if ses_ok_2 else 'NO'}")
print(f" [*] ¿Coinciden los P-valores al 100%?        -> {'SÍ' if pvals_ok_2 else 'NO'}")

print("\n--- Top 10 SNPs más significativos (Método 3D) ---")
indices_top = np.argsort(res_3d["p_val"])
snps_causales = [0, 1, 2, 3, 4]

for rank, idx in enumerate(indices_top[:10], 1):
    efecto = res_3d["beta"][idx]
    se = res_3d["SE"][idx]
    pval = res_3d["p_val"][idx]
    marcador = (
        " <<< (SNP CAUSAL SIMULADO)"
        if idx in snps_causales
        else " (Ruido de fondo)"
    )
    print(
        f" Rank {rank:02d}: SNP {idx:04d} | Beta: {efecto:+6.3f} | SE: {se:.3f} |"
        f" P-valor: {pval:.4e}{marcador}"
    )
