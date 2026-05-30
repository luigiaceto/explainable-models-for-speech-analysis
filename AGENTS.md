# PROJECT IDEA
This project, developed for an Explainable AI course of an MSc in Computer Engineering, consist in building an explainable-by-design model for speech analysis. In particular, a CBM for predicting emotion from audio traces. The architecture is the following:
- input: audio trace
- audio encoder (wav2vec2-base)
- audio embeddings extracted by the audio encoder
- pooling of the embeddings, obtaining a single audio embedding that encodes the audio informations of the entire trace
- perceptron layer
- CBM layer
- final classification layer

## Dataset to use
IEMOCAP, using only the audio utterances and their emotion labels. We ignore any video, text, and motion-capture modalities for model input and predict the full IEMOCAP emotion vocabulary.

## Concepts
Since the concepts are not annotated for the dataset, we should extract them. To do so, we have to employ python libraries to extract audio information from the traces: in this way we may create interpretable audio concepts to use as ground truth.

## Concept Bottleneck Layer
Using the predicted concept, the model has to predict the emotion of the audio.

## Loss
We aim at a good accuracy in both the concept prediction/regression task and emotion classification task. So we have to decide the loss to use in order to satisfy this requirement.

## Black-box model to compare with
We have to compare the CBM with a black box model, that is made similar to the CBM exception made for the concept bottleneck layer.

## Explainability metrics
Finally, we have to compute explainability metrics such as faithfulness

## Hardware resources
- MacBook Air M4
- Google Colab free

# PROJECT STRUCTURE IDEA
- data/ cotaining the dataset (it will be downloaded)
- data/features/ containing the embeddings (mean+std pooling) of the audio traces
- src/data/ containing the Dataset object
- checkpoints/ containing model weights
- src/models/ containing network architecture and custom losses
- src/preprocessing/ containing preprocessing scripts, etc.
- src/training/ containing the training scripts
- src/evaluation/ containing the evaluation scripts
- experiments_guide.ipynb is the notebook coordinating environment setup (pulling the repo, installing requirements, downloading the dataset), preprocessing, training, evaluation, etc.

# ARCHITECTURE IDEA

## Baseline blackbox
audio waveform
↓
wav2vec2-base frozen
↓
last_hidden_state: T × 768
↓
mean + std pooling
↓
embedding: 1536
↓
MLP classifier
↓
emotion prediction

In practice:

Input embedding: 1536

LayerNorm
Linear 1536 → 256
GELU / ReLU
Dropout 0.2

Linear 256 → 128
GELU / ReLU
Dropout 0.2

Linear 128 → 6

## CBM
audio waveform
↓
wav2vec2-base frozen
↓
last_hidden_state: T × 768
↓
mean + std pooling
↓
embedding: 1536
↓
concept predictor MLP
↓
continuous concept bottleneck: K concepts
↓
linear emotion classifier
↓
emotion prediction

In practice:

Input embedding: 1536

LayerNorm
Linear 1536 → 256
GELU / ReLU
Dropout 0.2

Linear 256 → 128
GELU / ReLU
Dropout 0.2

Linear 128 → K continuous concepts

Linear K → IEMOCAP emotions

# IMPORTANT
The entire project should be in English: filenames, variable names, comments, docs, etc.

# CHANGELOG
By now, only the black-box architecture has been implemented.
