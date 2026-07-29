import time
import numpy as np
from scipy import stats
import statsmodels.api as sm

# =====================================================================
# 1. DEFINICIÓN DE LAS TRES FUNCIONES (INTERFAZ COMÚN)
# =====================================================================


# FUNCIÓN DE REGRESIÓN LOGÍSTICA CON STATSMODELS (SNP por SNP)
def logreg_sm(
    matriz012: np.ndarray, phenotypes: np.ndarray, covariables: np.ndarray
) -> dict:
    """GWAS cualitativo (binario) utilizando statsmodels en un bucle clásico."""
    M, N = matriz012.shape
    if len(phenotypes) != N or covariables.shape[0] != N:
        raise ValueError("Error: El número de individuos de la matriz de genotipos no coincide con el de los fenotipos.")

    betas_SNP = np.empty(M)
    ses_SNP = np.empty(M)
    p_vals_SNP = np.empty(M)

    # Matriz de diseño (ceros en la columna de genotipos) Forma: N x K_total
    X = np.column_stack([np.ones(N), np.zeros(N), covariables])
    for i in range(M):
        # Añadimos la columna de ese SNP en la matriz de diseño
        X[:, 1] = matriz012[i, :]
        modelo = sm.Logit(phenotypes, X)
        resultados = modelo.fit(disp=0) # disp=0 evita que imprima el texto de convergencia en cada paso

        betas_SNP[i] = resultados.params[1]
        ses_SNP[i] = resultados.bse[1]
        p_vals_SNP[i] = resultados.pvalues[1]

    return {"beta": betas_SNP, "SE": ses_SNP, "p_val": p_vals_SNP}


# FUNCIÓN DE REGRESIÓN LOGÍSTICA CON IRLS EN NUMPY (SNP por SNP)
def logreg_nploop(
    matriz012: np.ndarray, phenotypes: np.ndarray, covariables: np.ndarray
) -> dict:
    """GWAS cualitativo implementando IRLS y Test de Wald en NumPy (bucle)."""
    M, N = matriz012.shape
    if len(phenotypes) != N or covariables.shape[0] != N:
        raise ValueError("Error: El número de individuos de la matriz de genotipos no coincide con el de los fenotipos.")

    betas_SNP = np.empty(M)
    ses_SNP = np.empty(M)
    p_vals_SNP = np.empty(M)

    # Matriz de diseño (ceros en la columna de genotipos) Forma: N x K_total
    X = np.column_stack([np.ones(N), np.zeros(N), covariables])
    for i in range(M):
        # Añadimos la columna de ese SNP en la matriz de diseño
        X[:, 1] = matriz012[i, :]
        K = X.shape[1]
        # Initial guess para Newton-Raphson
        beta = np.zeros(K)

        # Bucle IRLS de Newton-Raphson (máximo 15 iteraciones por seguridad)
        for _ in range(15):
            # Estimador lineal
            eta = X @ beta
            # Evitamos desbordamiento exponencial en la sigmoidea
            eta = np.clip(eta, -25, 25)

            # Probabilidades predichas de ser caso
            P = 1.0 / (1.0 + np.exp(-eta))
            # Vector de pesos W (varianza de Bernoulli)
            W = P * (1.0 - P)
            # Evitamos división por cero si P es exactamente 0 o 1
            W = np.clip(W, 1e-6, 1.0)

            # Primera y segunda derivada de Log-Verosimilitud. La Matriz de Información de Fisher I = - H (Matriz Hessiana)
            gradiente = X.T @ (phenotypes - P)
            Fisher = (X.T * W) @ X  # Equivalente a X.T @ np.diag(W) @ X pero más rápido

            # Actualización del paso de Newton: beta_new = beta_old + delta, donde delta = Fisher^-1 @ gradiente
            delta = np.linalg.solve(Fisher, gradiente)
            beta += delta

            if np.max(np.abs(delta)) < 1e-6:
                break

        # Al finalizar, la inversa de Fisher nos da la matriz de covarianza de los parámetros beta
        Fisher_inv = np.linalg.inv(Fisher)

        # Test de Wald (distribución Normal asintótica, no t-Student)
        betas_SNP[i] = beta[1]
        ses_SNP[i] = np.sqrt(Fisher_inv[1, 1])
        z_stat = betas_SNP[i] / ses_SNP[i]
        p_vals_SNP[i] = 2.0 * stats.norm.sf(np.abs(z_stat))

    return {"beta": betas_SNP, "SE": ses_SNP, "p_val": p_vals_SNP}


# FUNCIÓN DE REGRESIONES LOGÍSTICAS SIMULTÁNEAS (Array 3D)
def logreg_3d(
    matriz012: np.ndarray, phenotypes: np.ndarray, covariables: np.ndarray
) -> dict:
    """GWAS cualitativo vectorizado aplicando IRLS sobre tensores 3D en NumPy."""
    M, N = matriz012.shape
    if len(phenotypes) != N or covariables.shape[0] != N:
        raise ValueError("Error: El número de individuos de la matriz de genotipos no coincide con el de los fenotipos.")

    K_pcs = covariables.shape[1]
    K_total = K_pcs + 2

    # Construcción del tensor 3D de diseño (Forma: M x N x K_total)
    constante = np.ones((M, N, 1))
    snps_3d = matriz012[:, :, None]
    pcs_3d = np.broadcast_to(covariables, (M, N, K_pcs))
    X_3d = np.concatenate([constante, snps_3d, pcs_3d], axis=2)
    X_T = np.swapaxes(X_3d, 1, 2)   # Tensor transpuesto con forma: (M, K_total, N)

    # Initial guess para Newton-Raphson
    betas_all = np.zeros((M, K_total))

    # Algoritmo IRLS vectorizado para los M modelos en paralelo
    for _ in range(15):

        # Predictor lineal en 3D: sum(X * beta)
        eta = np.sum(X_3d * betas_all[:, None, :], axis=2)
        eta = np.clip(eta, -25, 25)

        P = 1.0 / (1.0 + np.exp(-eta))  # Forma: (M, N)
        W = P * (1.0 - P)
        W = np.clip(W, 1e-6, 1.0)       # Forma: (M, N)

        # Gradiente 3D: X^T @ (Y - P) para cada lote
        residuos = phenotypes - P
        gradiente = np.sum(X_T * residuos[:, None, :], axis=2)

        # Matriz de Fisher 3D: X^T @ W @ X para cada lote
        # Multiplicamos cada fila de X_T por los pesos W correspondientes
        X_T_W = X_T * W[:, None, :]     # Forma: (M, K_total, N)
        Fisher = X_T_W @ X_3d           # Forma: (M, K_total, K_total)

        # Ajuste de dimensión para resolver en 3D
        delta = np.linalg.solve(Fisher, gradiente[:, :, None])[:, :, 0]
        betas_all += delta

        if np.max(np.abs(delta)) < 1e-6:
            break

    # Extracción de betas y se. Test de Wald
    betas_SNP = betas_all[:, 1]
    Fisher_inv = np.linalg.inv(Fisher)
    ses_SNP = np.sqrt(Fisher_inv[:, 1, 1])

    z_stat = betas_SNP / ses_SNP
    p_vals_SNP = 2.0 * stats.norm.sf(np.abs(z_stat))

    return {"beta": betas_SNP, "SE": ses_SNP, "p_val": p_vals_SNP}


