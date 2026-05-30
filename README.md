# Explainable Models for Speech Analysis

This project studies speech emotion recognition on CREMA-D through a black-box
neural baseline and post-hoc prototype-based explanations.

At the moment, the implemented part of the project includes the black-box
baseline and a post-hoc prototype-based explanation method. In this repository,
the method is called **Post-hoc Latent Prototype Explainer (PLPE)**.

The black-box model is trained first, then its penultimate hidden representation
is extracted and used to build representative class prototypes. The method
explains a prediction by showing real training examples that are close to the
input in the representation space learned by the black-box classifier.

The current workflow is:

1. download the CREMA-D audio dataset;
2. extract frozen audio encoder embeddings, currently `microsoft/wavlm-large`;
3. apply mean + standard-deviation pooling;
4. train an MLP emotion classifier;
5. evaluate accuracy, macro F1, weighted F1, classification report, and confusion matrix;
6. extract L2-normalized black-box penultimate embeddings;
7. fit emotion-specific K-means clusters on those embeddings;
8. map each centroid to the nearest real training example of the same emotion;
9. evaluate the resulting prototype-based surrogate classifier;
10. measure surrogate fidelity against black-box predictions on the test split;
11. inspect individual predictions through their nearest latent prototypes.

The prototype component therefore has two roles:

- **post-hoc explainer**: retrieve representative training examples that the
  black-box maps close to the sample being explained;
- **prototype-based surrogate classifier**: predict emotions by summing
  similarities to the top-N prototypes and compare this surrogate against the
  ground-truth labels.

When the goal is to explain the black-box itself, the most relevant evaluation is
not only classification accuracy against the true labels, but also **fidelity**:
how often the prototype surrogate agrees with the black-box predictions.

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
- the Post-hoc Latent Prototype Explainer (PLPE), which fits K-means separately
  within each emotion class on the training split, maps each centroid to the
  nearest real training sample of the same emotion, selects K and top-N on the
  validation split, and evaluates once on the test split;
- a global fidelity utility that treats the black-box test predictions as the
  target labels and reports the prototype surrogate agreement accuracy;
- a per-sample prototype inspection utility that reports class scores, true
  label, predicted label, and the top-N real training prototypes used for one
  CREMA-D file name.

Additional explainability analyses beyond the current prototype-based workflow
are not implemented yet.

## Explanation Method

### Post-hoc Latent Prototype Explainer

PLPE explains the trained black-box in the space of its penultimate layer rather
than in the raw waveform space. The black-box classifier remains unchanged:

```text
audio waveform
-> frozen audio encoder
-> mean + std pooled embedding
-> MLP black-box
-> emotion prediction
```

After the black-box is trained, PLPE extracts the 128-dimensional activations
before the final classification layer:

```text
pooled audio embedding
-> trained black-box hidden layers
-> 128D latent representation
-> L2 normalization
```

The explainer then:

1. uses only the training split;
2. groups embeddings by emotion;
3. fits K-means separately inside each emotion class;
4. replaces each centroid with the nearest real training sample, so every
   prototype is listenable and inspectable;
5. classifies a new sample by retrieving its top-N nearest prototypes and
   summing cosine similarities per emotion.

This makes the explanation example-based: a prediction can be described as
"this sample is close, in the black-box latent space, to these representative
training samples".

### What PLPE Is Not

PLPE should not be described as a prototype network. Unlike architectures such
as ProtoPNet, prototypes are not part of the model's forward pass during
black-box training, and the black-box does not depend on them to make its
original prediction.

The method is better described as:

- a post-hoc prototype-based explainer;
- an example-based explanation method;
- a latent-space prototype analysis of the black-box;
- a prototype-based surrogate classifier for the black-box representation.

### Methodological Notes

The current implementation clusters samples by ground-truth emotion. This makes
the prototypes representative of dataset classes in the black-box latent space.
For a stricter black-box explanation setting, a useful extension is to cluster by
the black-box predicted class instead. That would make the prototypes represent
the behavior of the model, including systematic mistakes.

Useful evaluation metrics for PLPE include:

- classification accuracy, macro F1, and weighted F1 against ground-truth labels;
- global fidelity between the prototype surrogate predictions and black-box
  predictions, implemented as test-split agreement accuracy in
  `src/explainability/surrogate_fidelity.py`;
- local fidelity around individual samples;
- prototype purity within each emotion or predicted class;
- representativeness, measured by distances from samples to their nearest
  prototypes;
- stability of selected prototypes across random seeds and data splits.

## Related Work Context

Prototype-based deep learning includes methods where prototypes participate
directly in the learned model. For example, the original case-based prototype
network by Li et al. learns prototypes during training in an autoencoder latent
space, while ProtoPNet classifies images by comparing parts of an input with
learned prototypical parts. Those methods differ from PLPE because PLPE adds
prototypes only after the black-box has been trained.

There are also audio-specific prototype methods. "A Model You Can Hear" learns
playable spectral prototypes for audio identification. AudioProtoPNet adapts
ProtoPNet-style prototype reasoning to bird sound classification using
spectrogram embeddings. APEX, a recent audio XAI method, is closer in spirit to
PLPE because it is post-hoc and explains pre-trained audio classifiers through
audio prototypes.

Useful references:

- Li et al., "Deep Learning for Case-Based Reasoning through Prototypes:
  A Neural Network that Explains Its Predictions", 2017:
  https://arxiv.org/abs/1710.04806
- Chen et al., "This Looks Like That: Deep Learning for Interpretable Image
  Recognition", 2019:
  https://arxiv.org/abs/1806.10574
- Loiseau et al., "A Model You Can Hear: Audio Identification with Playable
  Prototypes", 2022:
  https://arxiv.org/abs/2208.03311
- Heinrich et al., "AudioProtoPNet: An interpretable deep learning model for
  bird sound classification", 2024:
  https://arxiv.org/abs/2404.10420
- Kawa et al., "APEX: Audio Prototype EXplanations for Classification Tasks",
  2026:
  https://arxiv.org/abs/2605.10153

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
- `src/explainability/`: utilities for prototype-score based inspection and
  surrogate fidelity evaluation.
- `src/utils/`: shared helper functions such as seed setup, device selection,
  and visualization utilities.

## Environment

If you use a CUDA GPU, install the PyTorch build that matches your CUDA version
before installing the remaining requirements.

For Google Colab, use `requirements-colab.txt` after cloning the repository in
the runtime.
