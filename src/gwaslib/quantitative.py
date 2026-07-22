# En este módulo definiré funciones para hacer la regresión lineal para caracteres cuantitativos
# import pynei as pn


# Primero, a partir de un vcf obtener la matriz 012


# Luego hacer un filtrado. Quedarnos con la matriz.


# Luego calcular un pca



# A groso modo, buscamos una función que tenga
# INPUTS: 
    # Una matriz de genotipos 012, ya filtrada (QCd) == Filas: SNPs   Columnas: Individuos
    # Una matriz (más bien vector) de fenotipos == Array tipo vector. Tantas entradas como individuos
    # Covariables (los PCs) La matriz calculada por el PCA de pynei. Filas: Individuos. Columnas: 10 PCs
    # Parámetros
        # Umbral arbitrario de p-valor para incluir SNPs en el output file (como en P-link)


# OUTPUTS

'''
def linear_reg(genotypes, phenotypes, covariates, params):

    # Para cada fila de la matriz012 (para cada SNP)
    for i in range(genotypes.shape[0]):

        # Hacer la regresión con sm.OLS (incluyendo los 10 primeros PCs como incógnitas)
        # Coger la fila de genotipos y "pasarla a una columna"
        X_gen = genotypes[i,:]

        # Añadir esa columna a la matriz de PCs (cada fila es un individuo y cada columna un PC, del 1º al 10º)
        X_total = np.hstack([X_gen.T, covariates])

        # Añadir la columna de unos (al principio o al final)

        # Con esa matriz coeficientes y con el vector fenotipos, hacer la regresión lineal

        # Calcular el estadístico t y el p-valor

        # Registrar el p-valor de ese SNP en el archivo output 

        '''


import numpy as np
from scipy import stats

import statsmodels.api as sm


# 1º FUNCIÓN DE REGRESIÓN CON STATSMODELS (Bucle for, SNP por SNP)
def gwas_quant_statsmodels(
    matriz012: np.ndarray, phenotypes: np.ndarray, covariables: np.ndarray
) -> dict:
  """Realiza un GWAS cuantitativo utilizando statsmodels en un bucle clásico."""
  M, N = matriz012.shape
  assert (
      len(phenotypes) == N and covariables.shape[0] == N
  ), "Error: El número de individuos no coincide."

  betas_SNP = np.empty(M)
  ses_SNP = np.empty(M)
  p_vals_SNP = np.empty(M)

  # Bucle línea por línea (SNP por SNP)
  for i in range(M):
    # 1. Matriz de diseño para 1 SNP (Forma: N x K_total)
    # Índice 0: Constante, Índice 1: SNP, Índices 2+: Covariables
    X = np.column_stack([np.ones(N), matriz012[i, :], covariables])
    # Vector de términos independientes (N, 1)
    Y = phenotypes

    # 2. Ajuste del modelo por Mínimos Cuadrados Ordinarios (OLS)
    modelo = sm.OLS(Y, X)
    resultados = modelo.fit()

    # 3. Extracción de los parámetros para el SNP (columna índice 1)
    betas_SNP[i] = resultados.params[1]
    ses_SNP[i] = resultados.bse[1]
    p_vals_SNP[i] = resultados.pvalues[1]

  return {"beta": betas_SNP, "SE": ses_SNP, "p_val": p_vals_SNP}

# 2º FUNCIÓN DE REGRESIÓN MANUALMENTE CON NUMPY (Bucle for, SNP por SNP)
def gwas_quant_bucle(
    matriz012: np.ndarray, phenotypes: np.ndarray, covariables: np.ndarray
) -> dict:
  """Realiza un GWAS cuantitativo utilizando un bucle clásico (modelo a

  modelo).
  """
  M, N = matriz012.shape
  assert (
      len(phenotypes) == N and covariables.shape[0] == N
  ), "Error: El número de individuos no coincide."

  K_total = covariables.shape[1] + 2  # Constante + SNP + Covariables
  df = N - K_total

  betas_SNP = np.empty(M)
  ses_SNP = np.empty(M)
  p_vals_SNP = np.empty(M)

  # Bucle línea por línea (SNP por SNP)
  for i in range(M):
    # 1. Matriz de diseño para 1 SNP (Forma: N x K_total)
    X = np.column_stack([np.ones(N), matriz012[i, :], covariables])
    # Vector de términos independientes (N, 1)
    Y = phenotypes

    # 2. OLS tradicional
    XTX = X.T @ X
    XTY = X.T @ Y
    beta_vec = np.linalg.solve(XTX, XTY)

    # 3. Residuos fenotípicos 
    residuos_feno = Y - (X @ beta_vec)
    sigma2_feno = np.sum(residuos_feno**2) / df

    # 4. Obtención del error estándar (SE) del SNP
    XTX_inv = np.linalg.inv(XTX)
    # El SNP siempre está en la columna índice 1
    betas_SNP[i] = beta_vec[1]
    ses_SNP[i] = np.sqrt(XTX_inv[1, 1] * sigma2_feno)

    # 5. Cálculo del estadístico y obtención del p-valor
    t_stat = betas_SNP[i] / ses_SNP[i]
    p_vals_SNP[i] = 2 * stats.t.sf(np.abs(t_stat), df=df)

  return {"beta": betas_SNP, "SE": ses_SNP, "p_val": p_vals_SNP}


# 3º FUNCIÓN DE REGRESIONES SIMULTÁNEAS (Array 3D)
def gwas_quant_3d(
    matriz012: np.ndarray, phenotypes: np.ndarray, covariables: np.ndarray
) -> dict:
  """Realiza un GWAS cuantitativo vectorizado utilizando tensores 3D en NumPy."""
  M, N = matriz012.shape
  assert (
      len(phenotypes) == N and covariables.shape[0] == N
  ), "Error: El número de individuos no coincide."

  K_pcs = covariables.shape[1]
  df = N - (K_pcs + 2)

  # 1. Matriz 3D de diseño (Forma: M x N x K_total)
  constante = np.ones((M, N, 1))
  snps_3d = matriz012[:, :, None]
  pcs_3d = np.broadcast_to(covariables, (M, N, K_pcs))
  X_3d = np.concatenate([constante, snps_3d, pcs_3d], axis=2)

  # Vector de términos independientes (N, 1)
  Y = phenotypes 

  # 2. Resolución matricial por lotes
  X_T = np.swapaxes(X_3d, 1, 2)
  XTX = X_T @ X_3d
  XTY_3d = (X_T @ Y)[:, :, None]  # Dimensión extra para compatibilidad 3D

  betas_all = np.linalg.solve(XTX, XTY_3d)[:, :, 0]
  betas_SNP = betas_all[:, 1]

  # 3. Residuos fenotípicos
  Y_pred = np.sum(X_3d * betas_all[:, None, :], axis=2)
  residuos_feno = Y - Y_pred
  sigma2_feno = np.sum(residuos_feno**2, axis=1) / df

  # 4. Obtención del error estándar (SE) de los SNP
  XTX_inv = np.linalg.inv(XTX)
  ses_SNP = np.sqrt(XTX_inv[:, 1, 1] * sigma2_feno)

  # 5. Cálculo del estadístico y obtención del p-valor
  t_stat = betas_SNP / ses_SNP
  p_vals_SNP = 2 * stats.t.sf(np.abs(t_stat), df=df)

  return {"beta": betas_SNP, "SE": ses_SNP, "p_val": p_vals_SNP}



# =====================================================================
# 2. GENERACIÓN DE DATOS SIMULADOS REALISTAS (20 SNPs, 200 Indivs, 10 PCs)
# =====================================================================
np.random.seed(12345)

M_snps = 20
N_indivs = 200
K_pcs = 10

print(
    f"--- 1. Generando datos coherentes: {M_snps} SNPs, {N_indivs} individuos,"
    f" {K_pcs} PCs ---"
)

# A. Genotipos realistas según Hardy-Weinberg (Frecuencias p entre 0.1 y 0.5)
p_alt = np.random.uniform(0.1, 0.5, size=(M_snps, 1))
prob_0 = (1 - p_alt) ** 2
prob_1 = 2 * p_alt * (1 - p_alt)
prob_2 = p_alt**2
probs = np.hstack([prob_0, prob_1, prob_2])

matriz012_test = np.empty((M_snps, N_indivs))
for idx_snp in range(M_snps):
  matriz012_test[idx_snp, :] = np.random.choice(
      [0, 1, 2], size=N_indivs, p=probs[idx_snp]
  )

# B. Covariables: PC1 refleja estructura poblacional (ancestría), PC2-PC10 ruido biológico
pc1 = np.random.normal(loc=0.0, scale=2.0, size=N_indivs)
pcs_resto = np.random.normal(loc=0.0, scale=1.0, size=(N_indivs, K_pcs - 1))
covariables_test = np.column_stack([pc1, pcs_resto])

# C. Fenotipo de altura (cm): Base + Efecto Ancestría (PC1) + Ruido ambiental
Y_test = (
    170.0
    + (3.0 * covariables_test[:, 0])
    + np.random.normal(loc=0.0, scale=4.0, size=N_indivs)
)

# D. INTRODUCIMOS EL ASOCIACIÓN REAL: Solo el SNP 0 tiene efecto (+4.0 cm por alelo)
# Los SNPs del 1 al 19 no tienen ningún efecto sobre Y
Y_test += 4.0 * matriz012_test[0, :]
Y_test = np.clip(Y_test, 140.0, 210.0)


# =====================================================================
# 3. EJECUCIÓN, TIEMPOS Y COMPROBACIÓN DE IDENTIDAD PARA EL TUTOR
# =====================================================================
import time

print("--- 2. Ejecutando regresión con bucle y STATSMODELS... ---")
t0 = time.perf_counter()
res_sm = gwas_quant_statsmodels(matriz012_test, Y_test, covariables_test)
t_sm = (time.perf_counter() - t0) * 1000

print("--- 3. Ejecutando regresión con bucle y NUMPY... ---")
t0 = time.perf_counter()
res_bucle = gwas_quant_bucle(matriz012_test, Y_test, covariables_test)
t_bucle = (time.perf_counter() - t0) * 1000

print("--- 4. Ejecutando regresión 3D vectorizada... ---")
t0 = time.perf_counter()
res_3d = gwas_quant_3d(matriz012_test, Y_test, covariables_test)
t_3d = (time.perf_counter() - t0) * 1000

print("\n====================================================================")
print(" COMPARATIVA DE TIEMPOS DE EJECUCIÓN (Para 20 SNPs)")
print("====================================================================")
print(f" -> Tiempo Statsmodels (bucle): {t_sm:.2f} ms")
print(f" -> Tiempo NumPy (bucle):       {t_bucle:.2f} ms")
print(f" -> Tiempo NumPy (3D):          {t_3d:.2f} ms")

print(
    f" [*] Aceleración Numpy (bucle) vs Statsmodels: {t_sm / t_bucle:.1f}x más rápido"
)
print(
    f" [*] Aceleración Numpy 3D vs Numpy (bucle): {t_bucle / t_3d:.1f}x más rápido"
)
print(
    f" [*] Aceleración Numpy 3D vs Statsmodels: {t_sm / t_3d:.1f}x más rápido"
)

print("\n====================================================================")
print(" VERIFICACIÓN DE IDENTIDAD MATEMÁTICA Numpy 3D vs Statsmodels (Tolerancia np.allclose)")
print("====================================================================")
# Comprobamos que el método 3D da exactamente lo mismo que Statsmodels
betas_ok = np.allclose(res_sm["beta"], res_3d["beta"])
ses_ok = np.allclose(res_sm["SE"], res_3d["SE"])
pvals_ok = np.allclose(res_sm["p_val"], res_3d["p_val"])

print(f" [*] ¿Coinciden las Betas al 100%?            -> {'SÍ' if betas_ok else 'NO'}")
print(f" [*] ¿Coinciden los Errores Estándar al 100%? -> {'SÍ' if ses_ok else 'NO'}")
print(f" [*] ¿Coinciden los P-valores al 100%?        -> {'SÍ' if pvals_ok else 'NO'}")

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
