# Explainable Models for Speech Analysis

This project studies speech emotion recognition on CREMA-D. The final goal is to
compare an explainable-by-design Concept Bottleneck Model (CBM) against a
black-box neural baseline.

At the moment, the implemented part of the project is the black-box baseline:

1. download the CREMA-D audio dataset;
2. extract frozen audio encoder embeddings, currently `microsoft/wavlm-base-plus`;
3. apply mean + standard-deviation pooling;
4. train an MLP emotion classifier;
5. evaluate accuracy, macro F1, weighted F1, classification report, and confusion matrix.

The workflow is coordinated from `experiments_guide.ipynb`.

The notebook defines the active audio encoder with a single tuple,
`FEATURE_EXTRACTOR = ("microsoft/wavlm-base-plus", 768)`, where the second
value is the encoder hidden-state size before pooling. The final MLP input size
is derived from the pooling method. Feature and checkpoint directories are also
derived from the model name, for example `data/features/wavlm_base_plus_mean_std/`
and `checkpoints/blackbox_wavlm_base_plus/`.

## Current Contents

The current codebase contains:

- a CREMA-D audio download pipeline based on an audio-only Hugging Face mirror;
- metadata parsing from CREMA-D filenames;
- dataset statistics utilities;
- frozen audio encoder feature extraction with masked mean + standard-deviation pooling;
- a PyTorch dataset for precomputed audio embeddings;
- an MLP black-box emotion classifier;
- a training loop with stratified train/validation/test split, weighted cross-entropy,
  AdamW, validation-based checkpointing, and early stopping;
- a separate evaluation pipeline for the saved black-box checkpoint;
- metric reporting utilities for accuracy, macro F1, weighted F1, per-class
  precision/recall/F1, predictions, and confusion matrix.

The CBM architecture, concept extraction, concept losses, and explainability
metrics are not implemented yet.

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
  extracting frozen audio encoder embeddings.
- `src/models/`: neural network definitions. Currently it contains the black-box
  MLP classifier.
- `src/training/`: training code. Currently it contains the black-box training
  loop and split creation logic.
- `src/evaluation/`: evaluation and metric reporting code for trained models.
- `src/utils/`: shared helper functions such as seed setup and device selection.

## Environment

If you use a CUDA GPU, install the PyTorch build that matches your CUDA version
before installing the remaining requirements.

For Google Colab, use `requirements-colab.txt` after cloning the repository in
the runtime.
