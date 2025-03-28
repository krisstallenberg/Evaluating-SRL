# Evaluating SRL

This repository contains code to run CheckList style tests on two SRL models, both trained on the English Universal Proposition Bank 1.0 CoNLL-U files.

## Installation

1. Clone this repository.
2. Download the [logistic regression and DistilBERT models](https://drive.google.com/drive/folders/1g4sd6abJWzzu58QtXJtqolxnwwxZAuQf?usp=share_link).
3. Place the full `learned-models` directory at the top-level of your local copy of this repository.
4. Create a new Python environment (use your preferred virtual environment tool, I used conda) e.g., `conda create -n challenge-srl`.
5. Activate your virtual environment, e.g. `conda activate challenge-srl`
6. Install the dependencies, e.g. `conda install --file requirements.txt`
    **Note**: This project uses [torch](https://pytorch.org/get-started/locally/), which is no longer available via conda. I recommend you install pip in your conda virtual environment (`conda install pip`) and install torch via pip: `pip3 install torch torchvision torchaudio`.
7. Download the Spacy **en_core_web_sm** model: `python -m spacy download en_core_web_sm`
8. Install an interactive python kernel in your virtual environment: `python -m ipykernel install --user --name=challenge-srl --display-name "Python (challenge-srl)"`


## Usage

1. Launch Jupyter notebook or lab, e.g. `jupyter lab`
2. Open `checklist-test.ipynb` and select the **Python (challenge-srl)** kernel.
3. Run all cells from top to bottom. The results of the evaluation are in `checklist-results.json` and displayed in the cell output.
