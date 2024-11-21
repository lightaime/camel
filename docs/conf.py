# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath('..'))

project = 'CAMEL'
copyright = '2024, CAMEL-AI.org'
author = 'CAMEL-AI.org'
release = '0.2.9'

html_favicon = (
    'https://raw.githubusercontent.com/camel-ai/camel/master/misc/favicon.png'
)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

source_suffix = ['.rst', '.md']


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinxawesome_theme',
    'myst_nb',
]

myst_enable_extensions = [
    "dollarmath",
    "colon_fence",
]
nb_execution_mode = "off"
nb_render_unexecuted_notebooks = False

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_permalinks_icon = "<span>^</span>"
html_theme = "sphinxawesome_theme"

# Update theme options for sphinx_awesome_theme
html_static_path = ['_static']

html_css_files = [
    'custom.css',
]
html_theme_options = {
    "logo_light": "_static/logo_primary_light.svg",
    "logo_dark": "_static/logo_primary_dark.svg",
}

nbsphinx_execute = 'never'
nbsphinx_allow_errors = True
nbsphinx_prolog = r"""
.. raw:: html

    <style>
    div.nbinput div.prompt,
    div.nboutput div.prompt {
        display: none;
    }
    </style>
"""
