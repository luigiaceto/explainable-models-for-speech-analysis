# Explainable Models for Speech Analysis

This project builds an explainable pipeline for speech emotion recognition. It combines a neural black-box classifier with a post-hoc, prototype-based explanation method that works in the representation space learned by the classifier.

The implemented explanation method is called **Post-hoc Latent Audio proTotype Organizer (PLATO)**. This name draws a deliberate parallel to Platonic philosophy, where individual instances in the observable world are understood and categorized based on their resemblance to pure, ideal forms. Similarly, PLATO explains the classification of an unseen test utterance by mapping it to the latent space and revealing the ideal, real-world audio prototypes that most closely embody its emotional characteristics.

A black-box model is trained first. Then its penultimate representation is extracted, normalized, clustered, and mapped back to real training utterances. A prediction can then be inspected by looking at the real audio examples that are closest to the sample in the black-box latent space.

The repository is organized around `experiments_guide.ipynb`, which coordinates the full experimental workflow: feature extraction, black-box training, black-box evaluation, latent embedding extraction, prototype construction, prototype-surrogate evaluation, fidelity analysis, and per-sample inspection.

## Current Workflow

The current pipeline is:

1. prepare the speech emotion dataset;
2. extract frozen audio encoder embeddings, with an audio encoder;
3. apply a pooling strategy to the encoder embeddings in order to obtain a single dense representation;
4. train an MLP black-box emotion classifier;
5. evaluate the black-box with accuracy, macro F1, weighted F1;
6. extract L2-normalized black-box penultimate embeddings;
7. fit class-specific K-means clusters in the black-box latent space;
8. map each centroid to the nearest real training sample of the same class;
9. classify samples with a prototype-based surrogate that sums similarities over all saved prototypes of each class;
10. evaluate the surrogate against ground-truth labels;
11. measure global fidelity against black-box predictions on the test split;
12. inspect individual predictions by printing class scores and listening to the closest prototype audio examples.

The prototype component has two roles:

- **post-hoc explainer**: retrieve representative training examples that the black-box maps close to the sample being explained;
- **prototype-based surrogate classifier**: classify samples by summing cosine similarities to all extracted prototypes of each class, then compare the resulting predictions against labels and black-box predictions.

When the goal is to explain the black-box itself, the key metric is not only classification performance against ground-truth labels, but also **fidelity**: how often the prototypes agrees with the black-box predictions.

## Dataset In This Branch

This branch uses **IEMOCAP Speech** in its preprocessed 4-class setup for speech emotion recognition. The implemented dataset utilities:

- download the `tarasabkar/IEMOCAP_Speech` Hugging Face mirror;
- write the WAV files under `data/raw/iemocap_4class/audio/`;
- normalize Hugging Face session split names such as `Session1` into session identifiers such as `Ses01`;
- store normalized metadata fields such as file name, session, emotion, integer label, audio path, and duration;
- map the four active emotions to integer labels: angry, happy, neutral, and sad;

| Class | Utterances | Notes |
| --- | --- | --- |
| **Neutral (Neutro)** | 1.708 | - |
| **Happy (Felice)** | 1.636 | Includes 595 of the original happy + 1.041 excited |
| **Angry (Rabbioso)** | 1.103 | — |
| **Sad (Triste)** | 1.084 | — |
| **Totale complessivo** | **5.531** | - |

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
- prototype-surrogate evaluation against ground-truth labels;
- global fidelity evaluation against black-box predictions;
- per-sample prototype inspection, including class scores, true label, predicted label, selected sample, and nearby prototype audio files;
- PCA visualizations for black-box embeddings and prototype overlays.

## Explanation Method

### Post-hoc Audio Latent Prototype Organizer

PLATO explains a trained black-box in the space of its penultimate layer rather than in the raw waveform space. The black-box classifier remains unchanged:

```text
audio waveform
-> frozen audio encoder
-> pooled embedding
-> MLP black-box
-> emotion prediction
```

After the black-box is trained, PLATO extracts the activations before the final classification layer:

```text
pooled audio embedding
-> trained black-box hidden layers
-> latent representation
-> L2 normalization
```

The explainer then:

1. uses only the training split to construct prototypes;
2. groups latent embeddings by ground-truth emotion;
3. fits K-means separately inside each emotion class;
4. replaces each centroid with the nearest real training sample, so every prototype is listenable and inspectable;
5. classifies a new sample by summing cosine similarities over all extracted prototypes of each emotion.

This makes the explanation example-based: a prediction can be described as "this sample is close, in the black-box latent space, to these representative training samples".

### What PLATO Is Not

PLATO should not be described as a prototype network. Unlike architectures such as ProtoPNet, prototypes are not part of the model's forward pass during black-box training, and the black-box does not depend on them to make its original prediction.

## Project Structure

- `experiments_guide.ipynb`: end-to-end notebook for running the experiment.
- `requirements.txt`: local Python dependencies.
- `requirements-colab.txt`: dependency set intended for Google Colab.
- `src/data/`: dataset adapters, metadata parsing, feature loading, statistics, and PyTorch dataset utilities.
- `src/preprocessing/`: dataset download, frozen audio feature extraction, and black-box latent embedding extraction.
- `src/models/`: black-box and prototype-surrogate model definitions.
- `src/training/`: black-box training, split creation, and prototype clustering grid search.
- `src/evaluation/`: metric computation and evaluation routines for trained models.
- `src/explainability/`: per-sample prototype inspection and surrogate fidelity utilities.
- `src/utils/`: naming, device selection, audio feature dimension helpers, and visualization utilities.
- `data/`: local raw data and generated feature matrices. Large generated artifacts are ignored by Git.
- `checkpoints/`: local model checkpoints, split files, training histories, and saved prototype files. This directory is ignored by Git.
- `reports/`: local evaluation outputs such as metrics, predictions, classification reports, confusion matrices, and PCA plots. This directory is ignored by Git.

## Related Work Context

Prototype-based deep learning includes methods where prototypes participate directly in the learned model. For example, the original case-based prototype network by Li et al. learns prototypes during training in an autoencoder latent space, while ProtoPNet classifies images by comparing parts of an input with learned prototypical parts. Those methods differ from PLATO because PLATO adds prototypes only after the black-box has been trained.

There are also audio-specific prototype methods. "A Model You Can Hear" learns playable spectral prototypes for audio identification. AudioProtoPNet adapts ProtoPNet-style prototype reasoning to bird sound classification using spectrogram embeddings. APEX is closer in spirit to PLATO because it is post-hoc and explains pre-trained audio classifiers through audio prototypes.

## Environment

If you use a CUDA GPU, install the PyTorch build that matches your CUDA version before installing the remaining requirements.
For Google Colab, install `requirements-colab.txt` after cloning the repository in the runtime (by running the cell n°0 of the notebook).