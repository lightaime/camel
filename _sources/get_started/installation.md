# Installation

## [Option 1] Install from PyPI
To install the base CAMEL library:
```bash
pip install camel-ai
```
Some features require extra dependencies:
- To install with all dependencies:
    ```bash
    pip install 'camel-ai[all]'
    ```
- To use the HuggingFace agents:
    ```bash
    pip install 'camel-ai[huggingface-agent]'
    ```
- To enable RAG or use agent memory:
    ```bash
    pip install 'camel-ai[tools]'
    ```

## [Option 2] Install from Source
### Install from Source with Poetry
```bash
# Make sure your python version is later than 3.10
# You can use pyenv to manage multiple python verisons in your sytstem

# Clone github repo
git clone https://github.com/camel-ai/camel.git

# Change directory into project directory
cd camel

# If you didn't install peotry before
pip install poetry  # (Optional)

# We suggest using python 3.10
poetry env use python3.10  # (Optional)

# Activate CAMEL virtual environment
poetry shell

# Install the base CAMEL library
# It takes about 90 seconds
poetry install

# Install CAMEL with all dependencies
poetry install -E all  # (Optional)

# Exit the virtual environment
exit
```

### Install from Source with Conda and Pip
```bash
# Create a conda virtual environment
conda create --name camel python=3.10

# Activate CAMEL conda environment
conda activate camel

# Clone github repo
git clone -b v0.2.22 https://github.com/camel-ai/camel.git

# Change directory into project directory
cd camel

# Install CAMEL from source
pip install -e .

# Or if you want to use all other extra packages
pip install -e '.[all]' # (Optional)
```
