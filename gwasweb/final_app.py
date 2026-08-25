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

    return


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



    #_subtitle_2 = mo.center(mo.md())
    _subtitle_3 = mo.center(mo.md('Nota: Los datos genotípicos deben subirse en un único fichero VCF. Los datos fenotípicos deben subirse en un único fichero .xlsx o .csv, con un formato de únicamente 2 columnas "Sample" y "Phenotype"'))


    _title = mo.center(mo.md(f"#**{_title_text}**"))
    _subtitle =  mo.center(mo.md(_subtitle_text))

    # Mostrar por pantalla el título y descripción:
    mo.vstack([_title, _subtitle])

    return


@app.cell
def _(mo):
    # Fabricar Instrucciones de uso
    _instruc_title_text = 'Instrucciones de uso'
    _instruc_text_1 = mo.md('**1º** Seleccionar tipo de fenotipo (cuantitativo o cualitativo)')
    _instruc_text_2 = mo.md('**2º** Subir los datos genotípicos y fenotípicos')
    _instruc_text_2_1 = mo.md('     - Los datos genotípicos deben subirse en un único fichero .vcf')
    _instruc_text_2_2 = mo.md('     - Los datos fenotípicos deben subirse en un único fichero .xslx o .csv')
    _instruc_text_2_3 = mo.md('**Nota:** El fichero de fenotipos debe contener únicamente 2 columnas: **"Sample" y "Phenotype"** (nombradas exactamente así)')
    _instruc_text_3 = mo.md('**3º** Seleccionar los parámetros de filtrado deseados')
    _instruc_text_4 = mo.md('**4º** Pulsar el botón "CORRER GWAS"')



    #, selecciona los parámetros deseados y ejecuta el análisis. Los resultados pueden visualizarse online y pueden ser descargados. **Los datos genotípicos deben subirse en un único fichero VCF. Los datos fenotípicos deben subirse en un único fichero .xlsx o .csv**, con un formato de únicamente 2 columnas "Sample" y "Phenotype"'

    _instruc_title = mo.md(f"**{_instruc_title_text}**")
    _instruc_subtitle = mo.vstack([_instruc_text_1, _instruc_text_2, _instruc_text_2_1, _instruc_text_2_2, _instruc_text_2_3, _instruc_text_3, _instruc_text_4])
    # Podemos hacer un mo.vstack de varios mo.md, pero no podemos hacer un mo.md de varios strings vstackeados.

    # Mostrarlo por pantalla
    mo.vstack([_instruc_title, _instruc_subtitle])
    return


@app.cell
def _(mo):
    # Fabricar dropdown para elegir tipo de fenotipo

    dropdown_quanti_quali = mo.ui.dropdown(options=['cuantitativo', 'cualitativo (binario)'], value='cuantitativo', label='Seleccione el tipo de fenotipo: ')

    dropdown_quanti_quali
    return


@app.cell
def _(mo):
    # Fabricar botón para subir CSV
    _label_vcf_button = 'Seleccione fichero (.vcf)'
    _previous_vcf_button = "Datos genotípicos de entrada: "

    csv_button_file = mo.ui.file(multiple=False, kind='button', label=_label_vcf_button, max_size = 100_000_000)
    show_csv_button_file =  mo.hstack([mo.md(_previous_vcf_button), csv_button_file], justify="start")

    show_csv_button_file
    return


@app.cell
def _(mo):
    # Fabricar botón para subir fenotipos .xlsx o .csv
    _label_pheno_button = 'Seleccione fichero (.xlsx o .csv)'
    _previous_pheno_button = "Datos fenotípicos de entrada: "

    pheno_button_file = mo.ui.file(multiple=False, kind='button', label=_label_pheno_button, max_size = 100_000_000)
    show_pheno_button_file =  mo.hstack([mo.md(_previous_pheno_button), pheno_button_file], justify="start")

    show_pheno_button_file
    return


if __name__ == "__main__":
    app.run()
