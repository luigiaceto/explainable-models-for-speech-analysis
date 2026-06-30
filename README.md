<img width="698" height="196" alt="Screenshot 2026-06-30 alle 22 40 28" src="https://github.com/user-attachments/assets/cca247cc-11f5-4c8d-b6a4-f8b6eadd4e42" />

# Explainable Models for Speech Analysis - PLATO

This project builds an explainable pipeline for speech emotion recognition (SER). It combines a neural black-box classifier with a post-hoc, prototype-based explanation method that works in the representation space learned by the classifier.

The implemented explanation method is called **Post-hoc Latent Audio proTotype Organizer (PLATO)**. A black-box model is trained first. Then its penultimate representation is extracted, normalized, clustered, and mapped back to real training utterances. A prediction can then be inspected by looking at the real audio examples that are closest to the sample in the black-box latent space.

The repository is organized around `experiments_guide.ipynb`, which coordinates the full experimental workflow: feature extraction, black-box training, black-box evaluation, latent embedding extraction, prototype construction, prototype-model evaluation, fidelity analysis, and per-sample inspection.

## XAI Pipeline

The XAI pipeline is:

1. prepare the speech emotion dataset;
2. extract frozen audio encoder embeddings, with an audio encoder;
3. apply a pooling strategy to the encoder embeddings in order to obtain a single dense representation;
4. train an MLP black-box emotion classifier;
5. evaluate the black-box with accuracy and macro F1;
6. extract L2-normalized black-box penultimate embeddings;
7. fit class-specific K-means clusters in the black-box latent space;
8. map each centroid to the nearest real training sample of the same class;
9. classify samples with a prototype-based model that sums similarities over all saved prototypes of each class;
10. evaluate the new model against ground-truth labels;
11. measure global fidelity against black-box predictions on the test split;
12. inspect individual predictions by printing class scores and listening to the closest prototype audio examples.

## Datasets

For each dataset, a separate project-copy (branch) has been created: *dev-IEMOCAP*, *dev-CREMAD*, *dev-TESS*. This is due to the fact that each dataset requires a different type of preprocessing and configuration. Overall, the majority of the code is the same for the 3 copies of the project.

### Datasets Description

**CREMA-D** for speech emotion recognition. The implemented dataset utilities:

- download an audio-only CREMA-D mirror from Hugging Face;
- write the WAV files under `data/raw/crema_d/AudioWAV/`;
- parse CREMA-D file names into metadata fields such as actor, sentence, emotion, and intensity;
- map the six emotions to integer labels;
- support both `sample_stratified` and `speaker_independent` train/validation and test splits.

**IEMOCAP** in its preprocessed 4-class setup for speech emotion recognition. The implemented dataset utilities:

- download the `tarasabkar/IEMOCAP_Speech` Hugging Face mirror;
- write the WAV files under `data/raw/iemocap_4class/audio/`;
- normalize Hugging Face session split names such as `Session1` into session identifiers such as `Ses01`;
- store normalized metadata fields such as file name, session, emotion, integer label, audio path, and duration;
- map the four active emotions to integer labels: angry, happy, neutral, and sad.

**TESS** for speech emotion recognition. The implemented dataset utilities:

- download the `AbstractTTS/TESS` Hugging Face mirror;
- write the WAV files under `data/raw/tess/audio/`;
- parse TESS file names into metadata fields such as speaker, speaker group, word, and emotion;
- normalize the seven emotions to the project vocabulary: angry, disgust, fear, happy, neutral, pleasant surprise, and sad;
- map the seven emotions to integer labels.

## Implemented Components

The codebase contains:

- dataset metadata parsing, feature loading, dataset statistics, and PyTorch dataset utilities;
- frozen audio encoder feature extraction with pooling;
- train/validation/test split creation;
- an MLP black-box classifier for precomputed audio features;
- black-box training with weighted cross-entropy, AdamW, optional plateau-based learning-rate scheduling, validation checkpointing, and early stopping;
- black-box evaluation utilities for metrics, predictions, classification reports, and confusion matrices;
- extraction of L2-normalized black-box penultimate embeddings;
- PLATO prototype construction through class-specific K-means clustering and centroid-to-real-sample mapping;
- prototype-model evaluation against ground-truth labels;
- global fidelity evaluation against black-box predictions;
- per-sample prototype inspection, including class scores, true label, predicted label, selected sample, and nearby prototype audio files;
- PCA visualizations for black-box embeddings and prototype overlays.

## Project Structure

- `experiments_guide.ipynb`: end-to-end notebook for running experiments.
- `requirements.txt`: local Python dependencies.
- `requirements-colab.txt`: dependency set intended for Google Colab.
- `src/data/`: dataset adapters, metadata parsing, feature loading, statistics, and PyTorch dataset utilities.
- `src/preprocessing/`: dataset download, frozen audio feature extraction, and black-box latent embedding extraction.
- `src/models/`: black-box and prototype-surrogate model definitions.
- `src/training/`: black-box training prototype clustering grid search.
- `src/evaluation/`: metric computation and evaluation routines for trained models.
- `src/explainability/`: per-sample prototype inspection and surrogate fidelity utilities.
- `src/utils/`: naming, device selection, audio feature dimension helpers, and visualization utilities.
- `data/`: local raw data and extracted features.
- `checkpoints/`: local model checkpoints, split files, training histories, and saved prototype files.
- `reports/`: local evaluation outputs such as metrics, predictions, classification reports, confusion matrices, and PCA plots.

## Environment

If you use a CUDA GPU, install the PyTorch build that matches your CUDA version before installing the remaining requirements.
For Google Colab, install `requirements-colab.txt` after cloning the repository in the runtime (by running the cell n°0 of the notebook).
