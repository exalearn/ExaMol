# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ExaMol'
copyright = '2023, Logan Ward'
author = 'Logan Ward'
html_title = 'ExaMol'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode'
]

intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
                       'mongoengine': ('https://docs.mongoengine.org/', None),
                       'parsl': ('https://parsl.readthedocs.io/en/stable/', None),
                       'colmena': ('https://colmena.readthedocs.io/en/stable/', None),
                       'proxystore': ('https://docs.proxystore.dev/main/', None)}
autodoc_mock_imports = ["tensorflow", "sklearn"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
