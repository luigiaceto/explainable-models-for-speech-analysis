# Explainable Models for Speech Analysis

This project studies speech emotion recognition on CREMA-D. The final goal is to
compare an explainable-by-design Concept Bottleneck Model (CBM) against a
black-box neural baseline.

At the moment, the implemented part of the project includes the black-box
baseline and a first prototype clustering classifier:

1. download the CREMA-D audio dataset;
2. extract frozen audio encoder embeddings, currently `microsoft/wavlm-large`;
3. apply mean + standard-deviation pooling;
4. train an MLP emotion classifier;
5. evaluate accuracy, macro F1, weighted F1, classification report, and confusion matrix;
6. extract L2-normalized black-box penultimate embeddings;
7. train and evaluate a prototype clustering classifier on those embeddings.

The workflow is coordinated from `experiments_guide.ipynb`.

The notebook defines the active audio encoder with a single tuple,
`FEATURE_EXTRACTOR = ("microsoft/wavlm-large", 1024)`, where the second
value is the encoder hidden-state size before pooling. The final MLP input size
is derived from the pooling method. Feature and checkpoint directories are also
derived from the model name, for example `data/features/wavlm_large_mean_std/`
and `checkpoints/blackbox_wavlm_large/`.

## Current Contents

The current codebase contains:

- a CREMA-D audio download pipeline based on an audio-only Hugging Face mirror;
- metadata parsing from CREMA-D filenames;
- dataset statistics utilities;
- frozen audio encoder feature extraction with masked mean + standard-deviation pooling;
- a PyTorch dataset for precomputed audio embeddings;
- an MLP black-box emotion classifier;
- a training loop with configurable train/validation/test split strategy
  (`sample_stratified` or `speaker_independent`), weighted cross-entropy,
  AdamW, optional plateau-based learning-rate scheduling, validation-based
  checkpointing, and early stopping;
- a separate evaluation pipeline for the saved black-box checkpoint;
- metric reporting utilities for accuracy, macro F1, weighted F1, per-class
  precision/recall/F1, predictions, and confusion matrix;
- extraction of 128-dimensional black-box penultimate embeddings with L2
  normalization;
- a prototype clustering classifier that fits K-means separately within each
  emotion class on the training split, maps each centroid to the nearest real
  training sample of the same emotion, selects K and top-N on the validation
  split, and evaluates once on the test split;
- a per-sample prototype inspection utility that reports class scores, true
  label, predicted label, and the top-N real training prototypes used for one
  CREMA-D file name.

The CBM architecture, concept extraction, concept losses, and full
explainability metrics are not implemented yet.

## Project Structure

- `data/`: local data storage. It is used for downloaded CREMA-D audio and
  generated feature matrices. This directory is ignored by Git because it can
  become large.
- `checkpoints/`: local model checkpoint storage. It stores trained model weights,
  split files, and training history. This directory is ignored by Git.
- `reports/`: local evaluation outputs such as test metrics, predictions, and
  confusion matrix plots. This directory is ignored by Git.
- `src/data/`: dataset-related code, including CREMA-D metadata parsing, class
  mappings, dataset statistics, feature loading, and the PyTorch feature dataset.
- `src/preprocessing/`: preprocessing code for downloading CREMA-D audio and
  extracting frozen audio encoder embeddings and black-box penultimate embeddings.
- `src/models/`: neural network definitions. Currently it contains the black-box
  MLP classifier and the prototype clustering classifier.
- `src/training/`: training code. Currently it contains the black-box training
  loop, split creation logic, and prototype clustering grid search.
- `src/evaluation/`: evaluation and metric reporting code for trained models.
- `src/explainability/`: utilities for prototype-score based inspection.
- `src/utils/`: shared helper functions such as seed setup, device selection,
  and visualization utilities.

## Environment

If you use a CUDA GPU, install the PyTorch build that matches your CUDA version
before installing the remaining requirements.

For Google Colab, use `requirements-colab.txt` after cloning the repository in
the runtime.
