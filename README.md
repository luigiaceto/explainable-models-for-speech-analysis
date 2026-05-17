# Explainable Models for Speech Analysis

This project studies speech emotion recognition on CREMA-D. The first implemented
pipeline is a black-box baseline:

1. download the CREMA-D audio dataset;
2. extract frozen `facebook/wav2vec2-base` embeddings;
3. apply mean + standard-deviation pooling;
4. train an MLP emotion classifier;
5. evaluate accuracy, macro F1, weighted F1, classification report, and confusion matrix.

The workflow is coordinated from `experiments_guide.ipynb`.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name speech-xai --display-name "Speech XAI"
```

If you use a CUDA GPU, install the PyTorch build that matches your CUDA version
before installing the remaining requirements.
