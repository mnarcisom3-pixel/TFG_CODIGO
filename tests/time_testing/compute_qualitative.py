'''
Este es un script donde se correrá la regresión logística mediante las 3 funciones creadas,
con tal de probar distintos tamaños de la matriz012, y distintos tamaños de efecto 

En ambos casos nos interesará ver
    - La viabilidad computacional
    - La aparición de falsos positivos
    - El poder para detectar true positives

Solo se correrá una vez cada método, así que los tiempos que aparecen no son promedios (no son fiables a solas)

'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import gwaslib.qualitative as gw_qual
import pynei

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

# A. Genotipos realistas según Hardy-Weinberg
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

# C. Fenotipo binario (0 = Control, 1 = Caso) basado en modelo logístico latente
# El SNP 0 tiene un efecto real (+1.5 en el log-odds por alelo alternativo)

# IMPORTANTE
# Se escogen (al azar) SNPs que no producen separación en este dataset simulado,
# para poder comparar el rendimiento de las tres implementaciones
# bajo condiciones en las que todas convergen correctamente.
eta_real = (
    -2.8
    + (1.6 * matriz012_test[0, :])
    - (1.4 * matriz012_test[1, :])
    + (1.5 * matriz012_test[98, :])
    - (1.5 * matriz012_test[3, :])
    + (1.3 * matriz012_test[4, :])
)
prob_real = 1.0 / (1.0 + np.exp(-eta_real))
phenotypes_test = np.random.binomial(n=1, p=prob_real)
# Esencialmente ha calculado la p(ser caso) de cada individuo según nuestro modelo (dando unos valores arbitrarios a las betas)
# Luego con la p de cada individuo, ha hecho una binomial de un solo intento, para ver si es caso o control con esa p

# =====================================================================
# 3. EJECUCIÓN, TIEMPOS Y COMPROBACIÓN DE IDENTIDAD PARA EL TUTOR
# =====================================================================

print("--- 2. Ejecutando regresión con bucle y STATSMODELS... ---")
t0 = time.perf_counter()
res_sm = gw_qual.logreg_sm(matriz012_test, phenotypes_test, covariables_test)
t_sm = (time.perf_counter() - t0) * 1000

print("--- 3. Ejecutando regresión con bucle y NUMPY... ---")
t0 = time.perf_counter()
res_bucle = gw_qual.logreg_nploop(matriz012_test, phenotypes_test, covariables_test)
t_bucle = (time.perf_counter() - t0) * 1000

print("--- 4. Ejecutando regresión 3D vectorizada... ---")
t0 = time.perf_counter()
res_3d = gw_qual.logreg_3d(matriz012_test, phenotypes_test, covariables_test)
t_3d = (time.perf_counter() - t0) * 1000

print("\n====================================================================")
print(" COMPARATIVA DE TIEMPOS DE EJECUCIÓN (Para 20 SNPs Logísticos)")
print("====================================================================")
print(f" -> Tiempo Statsmodels (bucle): {t_sm:.2f} ms")
print(f" -> Tiempo NumPy (bucle):       {t_bucle:.2f} ms")
print(f" -> Tiempo NumPy (3D):          {t_3d:.2f} ms")

print(
    f" [*] Aceleración Numpy (bucle) vs Statsmodels: {t_sm / t_bucle:.1f}x más rápido"
)
print(
    f" [*] Aceleración Numpy 3D vs Numpy (bucle):    {t_bucle / t_3d:.1f}x más rápido"
)
print(
    f" [*] Aceleración Numpy 3D vs Statsmodels:      {t_sm / t_3d:.1f}x más rápido"
)

print("\n====================================================================")
print(" VERIFICACIÓN DE IDENTIDAD MATEMÁTICA (Tolerancia estricta rtol=1e-5)")
print("====================================================================")
# Usamos rtol=1e-5 y atol=1e-12 para ser ultrarigurosos con los p-valores pequeños
betas_ok_1 = np.allclose(res_sm["beta"], res_bucle["beta"], rtol=1e-5, atol=1e-12)
ses_ok_1 = np.allclose(res_sm["SE"], res_bucle["SE"], rtol=1e-5, atol=1e-12)
pvals_ok_1 = np.allclose(res_sm["p_val"], res_bucle["p_val"], rtol=1e-5, atol=1e-12)

betas_ok_2 = np.allclose(res_bucle["beta"], res_3d["beta"], rtol=1e-5, atol=1e-12)
ses_ok_2 = np.allclose(res_bucle["SE"], res_3d["SE"], rtol=1e-5, atol=1e-12)
pvals_ok_2 = np.allclose(res_bucle["p_val"], res_3d["p_val"], rtol=1e-5, atol=1e-12)

print("Comparación entre logreg_sm y logreg_nploop:")
print(f" [*] ¿Coinciden las Betas al 100%?            -> {'SÍ' if betas_ok_1 else 'NO'}")
print(f" [*] ¿Coinciden los Errores Estándar al 100%? -> {'SÍ' if ses_ok_1 else 'NO'}")
print(f" [*] ¿Coinciden los P-valores al 100%?        -> {'SÍ' if pvals_ok_1 else 'NO'}")

print("\nComparación entre logreg_nploop y logreg_3d:")
print(f" [*] ¿Coinciden las Betas al 100%?            -> {'SÍ' if betas_ok_2 else 'NO'}")
print(f" [*] ¿Coinciden los Errores Estándar al 100%? -> {'SÍ' if ses_ok_2 else 'NO'}")
print(f" [*] ¿Coinciden los P-valores al 100%?        -> {'SÍ' if pvals_ok_2 else 'NO'}")

print("\n--- Top 3 SNPs más significativos (Método 3D) ---")
indices_top = np.argsort(res_3d["p_val"])
for rank, idx in enumerate(indices_top[:3], 1):
    efecto = res_3d["beta"][idx]
    se = res_3d["SE"][idx]
    pval = res_3d["p_val"][idx]
    marcador = (
        " <<< (SNP CAUSAL SIMULADO)" if idx == 0 else " (Ruido de fondo)"
    )
    print(
        f" Rank {rank}: SNP {idx:02d} | Beta: {efecto:+6.3f} | SE: {se:.3f} |"
        f" P-valor: {pval:.4e}{marcador}"
    )
