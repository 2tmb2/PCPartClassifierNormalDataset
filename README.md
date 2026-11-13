# PC Parts Image Classifier

This project classifies images of PC components into 11 categories using a
ResNet-18 image classifier.

The training script uses a pretrained ResNet-18 as a frozen feature extractor
and trains a final classification layer. It automatically uses CUDA, Apple
Metal (MPS), or CPU, depending on what is available.

## Requirements and setup

Use Python 3.12 or newer. From the repository root, create and activate a
virtual environment, then install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r src/requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.
The requirements file includes the libraries used by the training and
evaluation scripts, including PyTorch, torchvision, scikit-learn, matplotlib,
NumPy, and Pillow.

## Training data

The training images are included in this repository in `pc_parts/`. Keep the
folder at the repository root and organize it as one folder per class:

```text
pc_parts/
|-- cables/
|-- case/
|-- cpu/
|-- gpu/
|-- hdd/
|-- keyboard/
|-- monitor/
|-- motherboard/
|-- mouse/
|-- ram/
`-- ssd/
```

Place each category's image files in its corresponding folder. The folder
names are used as the class labels. The images currently in these folders are
from the [PC Parts Images Dataset Classification on Kaggle](https://www.kaggle.com/datasets/asaniczka/pc-parts-images-dataset-classification), with a large amount of cleaning as this dataset was clearly auto-scrapped and not manually collected.

## Run the classifier

Run commands from the repository root after activating the virtual
environment.

### Train a (neural network) model

```bash
python src/train_pc_parts.py
```

Training uses an 80/20 stratified train/validation split. The first run may
download the pretrained ResNet-18 weights if they are not already cached.
Model checkpoints, a reusable feature cache, and a training-loss plot are
written under `models/pc_parts/`.

### Open the prediction interface

```bash
python src/predict_gui.py
```

The application opens in your browser at `http://localhost:12345`. Select a
model, then upload an image or choose one of the sample images in
`uploaded_files/`. The interface lists checkpoints it finds under `models/`.
A trained checkpoint is already included under `models/pc_parts/`, so training
is not required just to try the interface.

### Evaluate the included model

```bash
python src/confusion.py
```

This evaluates the checkpoint configured in `src/confusion.py` against the
training/validation split and saves a confusion-matrix image under
`models/pc_parts/graphs/`.

## Other files

- `Baseline.ipynb` is an earlier baseline notebook.
- `src/hyperparameter_search.py` runs multiple training trials on many hyperparameters and writes results under `models/pc_parts/hyperparameter_search_results/`.
- `src/train_pca.py` and `src/train_svm.py` train models with a PCA and SVM instead of a fully connected layer. Run them the same way you would run train_pc_parts.py.
