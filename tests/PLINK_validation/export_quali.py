"""
Genera los datos de juguete cualitativos (fenotipo binario) y los exporta a
formato PLINK2. Necesita plink_io.py en el mismo directorio.
"""

import numpy as np
import pandas as pd

import gwaslib.qualitative as gw_qual
import pynei

from plink_io import export_to_plink

# ---------------------------------------------------------------------------
# 1. Generación de datos
# ---------------------------------------------------------------------------
np.random.seed(12345)

M_snps = 10000
N_indivs = 300
K_pcs = 10

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

mat012_forPCA = pd.DataFrame(matriz012_test.T)
PCA = pynei.pca.do_pca(mat012_forPCA)
covariables_test = PCA["projections"].to_numpy()[:, :10]

eta_real = (
    -2.8
    + (1.6 * matriz012_test[0, :])
    - (1.4 * matriz012_test[1, :])
    + (1.5 * matriz012_test[2, :])
    - (1.5 * matriz012_test[3, :])
    + (1.3 * matriz012_test[4, :])
)
prob_real = 1.0 / (1.0 + np.exp(-eta_real))
phenotypes_test = np.random.binomial(n=1, p=prob_real)

# ---------------------------------------------------------------------------
# 2. Exportar a formato PLINK
# ---------------------------------------------------------------------------
# Un único VCF y un único fichero de fenotipo sirven para los dos escenarios
# de validación; el --pca de PLINK se calcula sobre este mismo gwas_quali.vcf.
sample_ids = [f"IND{i + 1}" for i in range(N_indivs)]
export_to_plink(matriz012_test, phenotypes_test, covariables_test, sample_ids, "gwas_quali")

# ---------------------------------------------------------------------------
# 3. Guardar mis propios resultados para comparar después
# ---------------------------------------------------------------------------
own_results = gw_qual.logreg_3d(matriz012_test, phenotypes_test, covariables_test)
np.savez("own_results_quali.npz", **own_results)
