# ASD-Detection

This repository provides the main implementation used in our manuscript for audio-based ASD classification, including the proposed deep learning model, conventional machine learning baselines, and SHAP-based interpretability analysis.

## Repository Structure

- `AudioTransformer.py`  
  Implementation of the main Transformer-based audio classification model used in this study.

- `train_audio_transformer.py`  
  Training and evaluation script for the Transformer-based deep learning experiments.

- `train_machine_learning_model.py`  
  Script for the conventional machine learning baseline experiments.

- `ShapAnalysis.py`  
  Script for SHAP-based feature importance analysis on handcrafted acoustic features.

- `dataset.py`  
  Dataset construction and loading utilities used by the above experiments.

## Data Availability

Due to privacy and ethical restrictions associated with the human-subject audio dataset, the raw training and test audio files are not publicly released at this time.

## Minimal Example

A minimal forward-pass example is provided in `AudioTransformer.py` to illustrate the expected input format and model usage:

```python
import torch
from AudioTransformer import AudioClassifier

x = torch.randn(1, 1, 16000)
model = AudioClassifier()
y = model(x)
print(y.shape)
