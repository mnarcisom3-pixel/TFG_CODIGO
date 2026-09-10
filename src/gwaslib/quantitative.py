import numpy as np
from scipy import stats

import statsmodels.api as sm


# 1º FUNCIÓN DE REGRESIÓN CON STATSMODELS (Bucle for, SNP por SNP)
def linreg_sm(
    matriz012: np.ndarray, phenotypes: np.ndarray, covariables: np.ndarray
) -> dict:
    """Realiza un GWAS cuantitativo utilizando statsmodels en un bucle clásico."""
    M, N = matriz012.shape
    if len(phenotypes) != N or covariables.shape[0] != N:
        raise ValueError("Error: El número de individuos de la matriz de genotipos no coincide con el de los fenotipos.")

    betas_SNP = np.empty(M)
    ses_SNP = np.empty(M)
    p_vals_SNP = np.empty(M)

    # Matriz de diseño (ceros en la columna de genotipos) Forma: N x K_total
    X = np.column_stack([np.ones(N), np.zeros(N), covariables])
    # Bucle SNP por SNP
    for i in range(M):
        # Añadimos la columna de ese SNP en la matriz de diseño
        X[:, 1] = matriz012[i, :]
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
def linreg_nploop(
    matriz012: np.ndarray, phenotypes: np.ndarray, covariables: np.ndarray
) -> dict:
    """Realiza un GWAS cuantitativo utilizando un bucle clásico (modelo a

    modelo).
    """
    M, N = matriz012.shape
    if len(phenotypes) != N or covariables.shape[0] != N:
        raise ValueError("Error: El número de individuos de la matriz de genotipos no coincide con el de los fenotipos.")

    K_total = covariables.shape[1] + 2  # Constante + SNP + Covariables
    df = N - K_total

    betas_SNP = np.empty(M)
    ses_SNP = np.empty(M)
    p_vals_SNP = np.empty(M)

    # Matriz de diseño (con ceros en la columna de genotipos) Forma: N x K_total
    X = np.column_stack([np.ones(N), np.zeros(N), covariables])
    # Bucle SNP por SNP
    for i in range(M):
        # Añadimos la columna de ese SNP en la matriz de diseño
        X[:, 1] = matriz012[i, :]
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
def linreg_3d(
    matriz012: np.ndarray, phenotypes: np.ndarray, covariables: np.ndarray
) -> dict:
    """Realiza un GWAS cuantitativo vectorizado utilizando tensores 3D en NumPy."""
    M, N = matriz012.shape
    if len(phenotypes) != N or covariables.shape[0] != N:
        raise ValueError("Error: El número de individuos de la matriz de genotipos no coincide con el de los fenotipos.")

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
