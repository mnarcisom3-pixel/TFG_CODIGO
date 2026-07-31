"""
Genera los datos de juguete cuantitativos y los exporta a formato PLINK2.
Necesita plink_io.py en el mismo directorio (o en el PYTHONPATH).
"""

import numpy as np
import pandas as pd

import gwaslib.quantitative as gw_quant
import pynei

from plink_io import export_to_plink

# ---------------------------------------------------------------------------
# 1. Generación de datos (idéntico a tu script original)
# ---------------------------------------------------------------------------
np.random.seed(12345)

M_snps = 10000
N_indivs = 200
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

phenotypes_test = (
    170.0
    + (2.0 * matriz012_test[0, :])
    - (1.5 * matriz012_test[1, :])
    + (2.5 * matriz012_test[2, :])
    - (4.0 * matriz012_test[3, :])
    + (0.7 * matriz012_test[4, :])
    + np.random.normal(loc=0.0, scale=4.0, size=N_indivs)
)
phenotypes_test = np.clip(phenotypes_test, 140.0, 210.0)

# ---------------------------------------------------------------------------
# 2. Exportar a formato PLINK
# ---------------------------------------------------------------------------
# Un único VCF y un único fichero de fenotipo sirven para los dos escenarios
# de validación (mismos PCs / PCA propio de PLINK): lo único que cambia entre
# ambos es qué fichero de covariables se pasa a "--covar" en el --glm. Para
# el escenario "PCA propio de PLINK", el --pca de PLINK se ejecuta sobre este
# mismo gwas_quanti.vcf (no hace falta una versión sin covariables).
sample_ids = [f"IND{i + 1}" for i in range(N_indivs)]
export_to_plink(matriz012_test, phenotypes_test, covariables_test, sample_ids, "gwas_quanti")

# ---------------------------------------------------------------------------
# 3. Guardar mis propios resultados para comparar después
# ---------------------------------------------------------------------------

own_results = gw_quant.linreg_3d(matriz012_test, phenotypes_test, covariables_test)
np.savez("own_results_quanti.npz", **own_results)
