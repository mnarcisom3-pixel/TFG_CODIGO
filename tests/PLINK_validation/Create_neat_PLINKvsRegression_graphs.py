from pathlib import Path

# Importa las funciones desde el archivo donde las tengas definidas
from validation_utils import load_own_results, load_plink_glm, plot_comparison


# Directorio donde está este script
HERE = Path(__file__).parent

# Archivos de resultados
own_path = HERE / "own_results_quali.npz"
plink_path = HERE / "gwas_plink2_quali_ownpcs.PHENO.glm.logistic.hybrid"

# Cargar resultados
own_results = load_own_results(own_path)
plink_results = load_plink_glm(plink_path)

# Generar figura
plot_comparison(
    own=own_results,
    plink_df=plink_results,
    outfile=HERE / "validacion_gwas_cualitativo.png",
    causal_snp_idx=[0, 1, 2, 3, 4],
    beta_label="beta",
)