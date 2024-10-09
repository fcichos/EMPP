# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------

project = 'Einführung in die Modellierung Physikalischer Prozesse'
copyright = '2024, Frank Cichos'
author = 'Frank Cichos'
master_doc = 'index'
release = '24/25'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'nbsphinx',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    # Removed 'furo.sphinxext' as it's not a valid extension
]

exclude_patterns = ['_build', '**.ipynb_checkpoints']
templates_path = ['_templates']

mathjax3_config = {
    'TeX': {'equationNumbers': {'autoNumber': 'AMS', 'useLabelIds': True}},
}

# -- Options for HTML output -------------------------------------------------

html_theme = 'furo'

html_theme_options = {
    "source_repository": "https://github.com/fcichos/EMPP/",
    "source_branch": "main",
    "source_directory": "source/",
    "sidebar_hide_name": True,
}

html_last_updated_fmt = ""

html_static_path = ['_static']

html_logo = 'img/mona_logo.png'

def setup(app):
    app.add_css_file('theme_overrides.css')

# -- Extension configuration -------------------------------------------------

nbsphinx_prolog = r"""
{% set docname = env.doc2path(env.docname, base=False) %}

.. only:: html

    .. role:: raw-html(raw)
        :format: html

    .. nbinfo::
        This page was generated from `{{ docname }}`.
        :raw-html:`<br/><a href="https://colab.research.google.com/github/fcichos/EMPP/blob/main/source/{{ docname }}"><img alt="Colab badge" src="https://img.shields.io/badge/launch-%20colab-green.svg" style="vertical-align:text-bottom"></a>`
        :raw-html:`<br/><a href="https://mybinder.org/v2/gh/fcichos/EMPP.git/main?labpath=source/{{ docname }}"><img alt="Binder badge" src="https://img.shields.io/badge/launch-%20myBinder-red.svg" style="vertical-align:text-bottom"></a>`

.. only:: latex

    The following section was created from :file:`{{ docname }}`.
"""

nbsphinx_allow_errors = True

# Sphinx versioning settings

scv_show_banner = True
scv_whitelist_branches = ('main', 'master')