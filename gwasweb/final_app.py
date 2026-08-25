import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
async def _():
    # En WASM/Pyodide, instalamos dependencias de gwaslib y pynei (paquetes locales) antes de importarlos.
    # Localmente, estas dependencias están administradas por uv
    import sys

    if sys.platform == "emscripten":

        import micropip

        await micropip.install([
            "scipy",
            "statsmodels",
            "more-itertools",
            "matplotlib",
            "python-calamine",
            "pyarrow",
        ])

    # Importamos los paquetes locales una vez las dependencias WASM están disponibles
    import gwaslib as gw
    import pynei

    # Importamos los demás paquetes necesarios
    import io                           # for the uploads
    import tempfile                     # para leer el vcf con pynei
    from pathlib import Path            # used for tempfile
    import time                         # para barra de progreso en 

    import pandas
    import numpy
    import matplotlib.pyplot as plt

    return Path, gw, pynei, tempfile


@app.cell
def _():
    #----------------------------------------------------------------------------------------------------------------
    # Comenzamos definiendo los inputs del usuario: archivos y parámetros
    #----------------------------------------------------------------------------------------------------------------
    return


@app.cell
def _(mo):
    # Fabricar título y descripción de la web
    _title_text = 'GWASweb'
    _subtitle_text ='Esta es una web para ejecutar análisis GWAS en Python, basada en los paquetes **gwaslib** y **pynei**.'

    _title = mo.center(mo.md(f"#**{_title_text}**"))
    _subtitle =  mo.center(mo.md(_subtitle_text))

    # Mostrar por pantalla el título y descripción:
    mo.vstack([_title, _subtitle])
    return


@app.cell
def _(mo):
    # Fabricar instrucciones de uso

    # Pequeña función para escribir con un tamaño de letra menor
    def small_md(text):
        return mo.md(f'<span style="font-size:0.75em;">{text}</span>')

    # Todas las instrucciones, línea a línea
    _instruc_text_1 = small_md("<b>1º</b> Seleccionar tipo de fenotipo (cuantitativo o cualitativo)")

    _instruc_text_2 = small_md("<b>2º</b> Subir los datos genotípicos y fenotípicos")

    _instruc_text_2_1 = small_md("&nbsp;&nbsp;&nbsp;&nbsp;- Los datos genotípicos deben subirse en un único fichero <code>.vcf</code>")

    _instruc_text_2_2 = small_md("&nbsp;&nbsp;&nbsp;&nbsp;- Los datos fenotípicos deben subirse en un único fichero <code>.xlsx</code> o <code>.csv</code>")

    _instruc_text_2_3 = small_md('<b>Nota:</b> El fichero de fenotipos debe contener únicamente 2 columnas: <b>"Sample"</b> y <b>"Phenotype"</b> (nombradas exactamente así)')

    _instruc_text_3 = small_md("<b>3º</b> Seleccionar los parámetros de filtrado deseados")

    _instruc_text_4 = small_md('<b>4º</b> Pulsar el botón <b>"EJECUTAR GWAS"</b>')


    _instruc_subtitle = mo.vstack([
        _instruc_text_1,
        _instruc_text_2,
        _instruc_text_2_1,
        _instruc_text_2_2,
        _instruc_text_2_3,
        _instruc_text_3,
        _instruc_text_4,
    ])

    # Ponemos un desplegable con las instrucciones
    mo.accordion ({'####**Instrucciones de uso**': _instruc_subtitle})
    return (small_md,)


@app.cell
def _(mo):
    # Fabricar dropdown para elegir tipo de fenotipo

    dropdown_quanti_quali = mo.ui.dropdown(options=['cuantitativo', 'cualitativo (binario)'], value='cuantitativo', label='Seleccione el tipo de fenotipo: ')

    dropdown_quanti_quali
    return


@app.cell
def _(mo):
    # Fabricar botón para subir VCF
    _label_vcf_button = 'Seleccione fichero (.vcf)'
    _previous_vcf_button = "Datos genotípicos de entrada: "

    vcf_button_file = mo.ui.file(multiple=False, kind='button', label=_label_vcf_button, max_size = 100_000_000)
    show_vcf_button_file =  mo.hstack([mo.md(_previous_vcf_button), vcf_button_file], justify="start")

    show_vcf_button_file
    return (vcf_button_file,)


@app.cell
def _(mo):
    # Fabricar botón para subir fenotipos .xlsx o .csv
    _label_pheno_button = 'Seleccione fichero (.xlsx o .csv)'
    _previous_pheno_button = "Datos fenotípicos de entrada: "

    pheno_button_file = mo.ui.file(multiple=False, kind='button', label=_label_pheno_button, max_size = 100_000_000)
    show_pheno_button_file =  mo.hstack([mo.md(_previous_pheno_button), pheno_button_file], justify="start")

    show_pheno_button_file
    return (pheno_button_file,)


@app.cell
def _(Path, gw, pynei, tempfile):
    # Definimos funciones para poder cargar los datos de los ficheros subidos a los botones.

    # Para el VCF, queremos leerlo y crear un objeto Variants crudo con Pynei
    def load_uploaded_vcf(button_file):
        filename = button_file.value[0].name
        suffix = Path(filename).suffix.lower()

        if suffix not in {".vcf", ".gz"}:
            raise ValueError(
                "Genotype file must have '.vcf' or '.gz' extension."
            )

        # pynei.vars_from_vcf desde el disco, no la memoria, así que creamos un archivo temporal para leerlo
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(button_file.contents())  # Trasnferir el contenido al tempfile
            tmp_path = tmp.name

        return pynei.vars_from_vcf(tmp_path)   

    # Para el .csv o .xlsx de fenotipos, ya creamos una función load_phenotypes en gwaslib.integration.
    # Nos devuelve un pd.Series de fenotipos crudos.

    # Aquí simplemente hemos de crear una función que pueda trabajar con el button_file de marimo
    def load_uploaded_phenotypes(button_file):
        filename = button_file.value[0].name

        return gw.load_phenotypes(
            button_file.contents(),
            filename=filename,
        )

    return load_uploaded_phenotypes, load_uploaded_vcf


@app.cell
def _(load_uploaded_vcf, mo, vcf_button_file):
    # En el momento en el que se suba un archivo a un botón, lo leemos
    # Para el fichero de GENOTIPOS
    variants_crude = None
    vcf_error = None

    if vcf_button_file.value:
        try:
            variants_crude = load_uploaded_vcf(vcf_button_file)
        except (ValueError, KeyError) as e:
            vcf_error = mo.md(str(e)).callout(kind="danger")

    vcf_error
    return


@app.cell
def _(load_uploaded_phenotypes, mo, pheno_button_file):
    # Para el fichero de FENOTIPOS
    phenotypes_crude = None
    pheno_error = None

    if pheno_button_file.value:
        try:
            phenotypes_crude = load_uploaded_phenotypes(pheno_button_file)
        except (ValueError, KeyError, TypeError) as e:
            pheno_error = mo.md(str(e)).callout(kind="danger")

    pheno_error
    return


@app.cell
def _(mo, small_md):
    # Fabricar los sliders para parámetros de filtrado


    slider_max_sample_missing = mo.ui.slider(start=0, stop=0.3, step=0.01, value=0.05, include_input=True, label='Máximo % de genotipos faltantes permitido por individuo')
    slider_max_snp_missing = mo.ui.slider(start=0, stop=0.3, step=0.01, value=0.05, include_input=True, label='Máximo % de genotipos faltantes permitido por SNP')
    slider_min_maf = mo.ui.slider(start=0.05, stop=0.50, step=0.01, value=0.05, include_input=True, label='Frecuencia mínima del alelo no mayoritario')

    NaN_disclaimer = small_md(
        """
        <b>Nota</b>: Además de este filtro, se eliminan también todos los individuos/muestras cuyo valor del <b>fenotipo</b> sea desconocido (NaN).
        """
    )
    LD_disclaimer = small_md(
        """
        <b>Nota</b>: Además de estos filtros, <b>en el caso del PCA se incluye un cribado de SNPs por desequilibrio de ligamiento (LD)</b>.
        Esto se hace para evitar que regiones con LD elevado ejerzan una influencia desproporcionada sobre los componentes principales.
        <b>Este cribado no se aplica en el filtrado para GWAS</b>, ya que se busca maximizar la cobertura genómica para encontrar asociaciones.
        """
    )

    sample_parameters = mo.vstack([mo.md('**Filtrado de individuos/muestras**'), slider_max_sample_missing, NaN_disclaimer])
    snp_parameters = mo.vstack([mo.md('**Filtrado de SNPs**'),  slider_max_snp_missing, slider_min_maf, LD_disclaimer])

    parameters = mo.accordion({"####**Parámetros de filtrado** (comunes para el PCA y el GWAS): ": mo.vstack([sample_parameters, mo.Html("<div style='height:20px'></div>"), snp_parameters])})

    parameters
    return


if __name__ == "__main__":
    app.run()
