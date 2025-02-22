# How to update the documentation

To update the RST files:
```bash
sphinx-apidoc -o docs camel/
```

Helpful article [here](https://towardsdatascience.com/documenting-python-code-with-sphinx-554e1d6c4f6d).

# Building Documentation

To build the documentation:

1. [Install CAMEL](https://github.com/camel-ai/camel/blob/master/README.md) from source.

2. Install required dependencies running the following command in your terminal or command prompt:
```bash
pip install sphinx
pip install sphinx_book_theme
pip install sphinx-autobuild
pip install myst_parser
pip install nbsphinx
```

3. Build the document and launch the HTML documentation.
```bash
cd docs
sphinx-autobuild . _build/html --port 8000
```

This command starts a local HTTP server on port 8080 by default. Once the server is running, open your web browser and enter the following URL:
```bash
127.0.0.1:8000
```
This will load the HTML documentation in your web browser from the local server. The server will watch for changes in your source files and automatically rebuild the documentation and refresh the page in the browser when you make changes – so changes in docs will be immediately reflected in the rendered doc.

You can navigate through the documentation using the links and interact with it as you would with any other web page.

To stop the local server, go back to the terminal or command prompt where it is running and press `Ctrl+C` to terminate the server.

In case the autobuild does not work, you may use the traditional build approach:
```bash
cd docs
make html
cd _build/html
python -m http.server
```
