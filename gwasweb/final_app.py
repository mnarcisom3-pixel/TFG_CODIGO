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

    return Path, gw, io, pynei, tempfile, time


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

    _instruc_text_2_3 = small_md('<b>Nota:</b> El fichero de fenotipos debe contener únicamente 2 columnas: <b>"Sample"</b> y <b>"Phenotype"</b> (nombradas exactamente así). Para <b>fenotipos cualitativos (binarios)</b>, la columna "Phenotype" solo debe contener <b>valores de 0 y 1</b>')

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
    return (dropdown_quanti_quali,)


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
    return (variants_crude,)


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
    return (phenotypes_crude,)


@app.cell
def _(mo, small_md):
    # Fabricar los sliders para parámetros de filtrado
    slider_max_sample_missing = mo.ui.slider(start=0, stop=0.2, step=0.01, value=0.05, include_input=True, label='Máximo % de genotipos faltantes permitido por individuo')
    slider_max_snp_missing = mo.ui.slider(start=0, stop=0.2, step=0.01, value=0.05, include_input=True, label='Máximo % de genotipos faltantes permitido por SNP')
    slider_max_maf = mo.ui.slider(start=0.80, stop=0.95, step=0.01, value=0.95, include_input=True, label='Máxima frecuencia permitida para el alelo mayoritario')

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
    snp_parameters = mo.vstack([mo.md('**Filtrado de SNPs**'),  slider_max_snp_missing, slider_max_maf, LD_disclaimer])

    parameters = mo.accordion({"####**Parámetros de filtrado** (comunes para el PCA y el GWAS): ": mo.vstack([sample_parameters, mo.Html("<div style='height:20px'></div>"), snp_parameters])})

    parameters
    return slider_max_maf, slider_max_sample_missing, slider_max_snp_missing


@app.cell
def _(mo, small_md):
    # Fabricar desplegable para los parámetros de los gráficos
    graph_title_pheno_name = mo.ui.text(label="Nombre del fenotipo para mostrar en las gráficas",placeholder="Ej. Plant height")

    dropdown_manhattan_y_axis = mo.ui.dropdown(options=['-log10(p-valores) crudos (opción recomendada)', '-log10(p-valores) corregidos por Bonferroni', '-log10(p-valores) corregidos por Benjamini-Hochberg FDR'], value='-log10(p-valores) crudos (opción recomendada)', label='Variable para mostrar en el eje Y del Manhattan plot: ')

    options_for_Manhattan_function = ["p", "bonferroni", "fdr"]

    y_axis_options_dict = dict(
        zip(dropdown_manhattan_y_axis.options, options_for_Manhattan_function)
    )


    parameters_graph = mo.accordion({"####**Parámetros para crear las gráficas de resultados** (Manhattan Plot y QQ-Plot): ": mo.vstack([graph_title_pheno_name, dropdown_manhattan_y_axis, small_md("<b>Nota:</b> Aunque se seleccionen los p-valores crudos, <b>todos los Manhattan plots incluyen un umbral de significancia corregido por múltiples tests</b>")])})

    parameters_graph
    return (
        dropdown_manhattan_y_axis,
        graph_title_pheno_name,
        y_axis_options_dict,
    )


@app.cell
def _(mo):
    # Creamos botón para ejecutar GWAS
    color_run_gwas = 'success'

    run_gwas_button = mo.ui.run_button(label='EJECUTAR GWAS', kind=color_run_gwas, full_width=False, tooltip='Haga click para iniciar el análisis')
    run_gwas_button
    return (run_gwas_button,)


@app.cell
def _(
    dropdown_quanti_quali,
    gw,
    mo,
    phenotypes_crude,
    pynei,
    run_gwas_button,
    slider_max_maf,
    slider_max_sample_missing,
    slider_max_snp_missing,
    time,
    variants_crude,
):
    # Una vez pulsado el botón, ejecutamos el análisis
    gwas_results = None
    df_all_pcs = None
    analysis_error = None
    analysis_elapsed_time = None
    analysis_stage_times = None


    if run_gwas_button.value:

        # Comprobar que los dos archivos se han cargado correctamente
        if variants_crude is None or phenotypes_crude is None:

            analysis_error = mo.callout(
                "Debe cargar correctamente los datos genotípicos y fenotípicos "
                "antes de ejecutar el análisis.",
                kind="danger",
            )

        else:

            _total_start = time.perf_counter()
            analysis_stage_times = {}  # Diccionario para almacenar el tiempo que ha tardado cada paso

            try:

                # =========================================================
                # STEP 1 — PREPARAR DATOS
                # =========================================================

                _t0 = time.perf_counter()

                with mo.status.spinner(
                    title="PASO 1/3 — Preparando los datos para el análisis..."
                ):
                    _filtered_vars_for_GWAS = gw.filter_genotypes_for_GWAS(
                        variants=variants_crude,
                        phenotypes=phenotypes_crude,
                        max_sample_gt_missing_rate=slider_max_sample_missing.value,
                        max_var_gt_missing_rate=slider_max_snp_missing.value,
                        max_allowed_maf=slider_max_maf.value,
                    )

                    _filtered_pheno = gw.filter_phenotypes(
                        phenotypes_crude,
                        _filtered_vars_for_GWAS.samples,
                    )

                analysis_stage_times["Preparación de datos"] = (time.perf_counter() - _t0)


                # =========================================================
                # STEP 2 — FILTRADO PCA + PCA
                # =========================================================

                _t0 = time.perf_counter()

                with mo.status.spinner(
                    title="PASO 2/3 — Filtrando datos para PCA y ejecutando PCA... (esto podría tardar entre minutos y horas, dependiendo del VCF subido)"
                ):
                    _filtered_vars_for_PCA = gw.filter_genotypes_for_PCA(
                        variants=variants_crude,
                        phenotypes=phenotypes_crude,
                        max_sample_gt_missing_rate=slider_max_sample_missing.value,
                        max_var_gt_missing_rate=slider_max_snp_missing.value,
                        max_allowed_maf=slider_max_maf.value,
                        min_allowed_r2=0.1,
                    )

                    _pca_dict = pynei.pca.do_pca_with_vars(_filtered_vars_for_PCA,transform_to_biallelic=True)

                    df_all_pcs = _pca_dict["projections"]

                analysis_stage_times["Filtrado PCA + PCA"] = (time.perf_counter() - _t0)


                # =========================================================
                # STEP 3 — FILTRADO GWAS + GWAS
                # =========================================================

                _t0 = time.perf_counter()

                with mo.status.spinner(
                    title="PASO 3/3 — Filtrando datos para GWAS y ejecutando GWAS..."
                ):
                    gwas_results = gw.do_gwas(
                        filtered_vars=_filtered_vars_for_GWAS,
                        filtered_phenotypes=_filtered_pheno,
                        covariates=df_all_pcs,
                        type_of_phenotype=dropdown_quanti_quali.selected_key,
                        sort_by_significance=False,
                    )

                analysis_stage_times["GWAS"] = (time.perf_counter() - _t0)

                # Tiempo total del análisis
                analysis_elapsed_time = (time.perf_counter() - _total_start)


            except Exception as _e:

                analysis_elapsed_time = (time.perf_counter() - _total_start)

                gwas_results = None

                analysis_error = mo.callout(
                    mo.md(
                        f"""
                        **El análisis no pudo completarse.**

                        `{type(_e).__name__}: {_e}`

                        Revise los datos de entrada y los parámetros de filtrado.
                        """
                    ),
                    kind="danger",
                )

    # =========================================================
    # OUTPUT DE LA CELDA
    # =========================================================

    if analysis_error is not None:
        _analysis_output = analysis_error

    elif gwas_results is not None:
        _analysis_output = mo.callout(
            mo.md(
                f"""
                **Análisis completado correctamente**

                *Tiempos de computación*
                - 1. Preparación de datos: `{analysis_stage_times["Preparación de datos"]:.2f} s`
                - 2. Filtrado PCA + PCA: `{analysis_stage_times["Filtrado PCA + PCA"]:.2f} s`
                - 3. Filtrado GWAS + GWAS: `{analysis_stage_times["GWAS"]:.2f} s`
                - **Tiempo total: `{analysis_elapsed_time:.2f} s`**
                """
            ),
            kind="success",
        )

    else:
        _analysis_output = mo.md("")

    # Muy importante: en los ifs, definimos el output. Pero al acabar, hemos de mostrarlo por pantalla así
    _analysis_output
    return df_all_pcs, gwas_results


@app.cell
def _(
    df_all_pcs,
    dropdown_manhattan_y_axis,
    graph_title_pheno_name,
    gw,
    gwas_results,
    io,
    mo,
    y_axis_options_dict,
):
    # Crear las gráficas a partir de los resultados
    fig_pca = None
    fig_manhattan = None
    fig_qq = None

    download_pca = None
    download_manhattan = None
    download_qq = None

    # Función para que el usuario pueda descargar las gráficas también
    def figure_to_png(fig):
        _buffer = io.BytesIO()

        fig.savefig(
            _buffer,
            format="png",
            dpi=300,
            bbox_inches="tight",
        )

        return _buffer.getvalue()


    if gwas_results is not None and df_all_pcs is not None:

        # =========================================================
        # PCA plot
        # =========================================================

        fig_pca, _ = gw.create_pca_plot(df_all_pcs)

        # =========================================================
        # Phenotype name
        # =========================================================

        _phenotype_name = (graph_title_pheno_name.value.strip() or "Phenotype")

        # Para el nombre del archivo de la gráfica descargado
        _safe_phenotype_name = "_".join(_phenotype_name.strip().lower().split())

        # =========================================================
        # Manhattan plot
        # =========================================================

        fig_manhattan, _ = gw.create_manhattan_plot(
            gwas_results,
            y_axis_variable=y_axis_options_dict[dropdown_manhattan_y_axis.value],
            phenotype_name=_phenotype_name,
        )

        # =========================================================
        # QQ plot
        # =========================================================

        fig_qq, _ = gw.create_qq_plot(
            gwas_results,
            phenotype_name=_phenotype_name,
        )

        # =========================================================
        # Download buttons
        # =========================================================

        download_pca = mo.download(
            data=lambda: figure_to_png(fig_pca),
            filename=f"PCA_plot_{_safe_phenotype_name}.png",
            mimetype="image/png",
            label="Descargar PNG",
        )

        download_manhattan = mo.download(
            data=lambda: figure_to_png(fig_manhattan),
            filename=f"Manhattan_plot_{_safe_phenotype_name}.png",
            mimetype="image/png",
            label="Descargar PNG",
        )

        download_qq = mo.download(
            data=lambda: figure_to_png(fig_qq),
            filename=f"QQ_plot_{_safe_phenotype_name}.png",
            mimetype="image/png",
            label="Descargar PNG",
        )
    return (
        download_manhattan,
        download_pca,
        download_qq,
        fig_manhattan,
        fig_pca,
        fig_qq,
    )


@app.cell
def _(
    download_manhattan,
    download_pca,
    download_qq,
    fig_manhattan,
    fig_pca,
    fig_qq,
    gwas_results,
    mo,
):
    # Mostrar los resultados del análisis en la web
    if gwas_results is not None:

        _results_tabs = mo.ui.tabs({

            "GWAS results": gwas_results,
            "PCA plot": mo.vstack([fig_pca, download_pca]),
            "Manhattan plot": mo.vstack([fig_manhattan, download_manhattan]),
            "QQ plot": mo.vstack([fig_qq, download_qq]),
        })

        _results_output = mo.vstack([mo.md("## Resultados"), _results_tabs])

    else:
        _results_output = mo.md("")

    _results_output
    return


if __name__ == "__main__":
    app.run()
