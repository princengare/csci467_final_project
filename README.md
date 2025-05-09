# Analyzing the Language of Love Across Cultures

## Introduction

Love is a deeply human experience, and its expression varies widely across languages, cultures, and platforms. In this project, we explore the linguistic patterns of love using diverse datasets drawn from poetry, music lyrics, literature, tweets, and multilingual dialogue. Our goal is to classify types of love (romantic, familial, platonic) and estimate emotional dimensions such as valence, arousal, and dominance (VAD) using modern machine learning models.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Related Work and Methodologies](#related-work-and-methodologies)
  - [Avoiding Overfitting](#avoiding-overfitting)
  - [Multilingual and Multimodal Love Detection](#multilingual-and-multimodal-love-detection)
  - [Model Hyperparameter Optimization](#model-hyperparameter-optimization)
- [Dataset and Evaluation](#dataset-and-evaluation)
  - [Dataset Sources](#dataset-sources)
  - [Dataset Split](#dataset-split)
  - [Evaluation Metrics](#evaluation-metrics)
- [Methods](#methods)
  - [Baseline](#baseline)
  - [XLM-R (Frozen)](#xlm-r-frozen)
  - [XLM-R (Fine-Tuned)](#xlm-r-fine-tuned)
  - [VAD Regression](#vad-regression)
- [Results](#results)
- [Conclusion](#conclusion)
- [Notes for macOS/M1 Users](#notes-for-macosm1-users)
- [References](#references)

## Requirements

Clone the repository and install required packages:

```bash
git clone https://github.com/princengare/csci467-final-project.git
cd csci467-final-project
pip install -r requirements.txt
```
You will need to download the English-French OPUS corpus manually or script it as part of the collect_datasets.py routine. Put it under /opus.

Then run the full pipeline:

```bash
python3 download_gutenberg_love_texts.py
python3 collect_datasets.py
python3 train_model.py
```

## Related Work and Methodologies

### Avoiding Overfitting

- Stratified data splits were used to maintain class balance.
- Early stopping was applied during fine-tuning to prevent overfitting.

### Multilingual and Multimodal Love Detection

We use XLM-RoBERTa, a multilingual transformer, to classify affective text from multiple domains — poetry, tweets, lyrics, and literature.

### Model Hyperparameter Optimization

- Batch size: 16  
- Learning rate: 2e-5  
- Epochs: 4  
- Token max length: 256

## Dataset and Evaluation

### Dataset Sources

| Source            | Description                                              |
|-------------------|----------------------------------------------------------|
| Poetry Foundation | Romantic poems scraped from the [Poetry Foundation](https://www.poetryfoundation.org/) site |
| Genius Lyrics     | Songs by Adele, Ed Sheeran, and Taylor Swift             |
| Twitter           | Tweets using #LoveYou, filtered for English              |
| Project Gutenberg | English & French romantic passages ([Link](https://www.gutenberg.org/)) |
| OPUS Dialogue     | English-French subtitle lines containing "love" ([Link](https://opus.nlpl.eu/)) |

### Dataset Split

- Train: 70%  
- Dev: 15%  
- Test: 15%

### Evaluation Metrics

- Accuracy  
- F1 Score (macro average)  
- Mean Absolute Error (MAE) for VAD

## Methods

### Baseline

A rule-based keyword classifier using hand-crafted lists for romantic, platonic, and familial expressions.

### XLM-R (Frozen)

Evaluates the XLM-RoBERTa model without fine-tuning as a zero-shot multilingual classifier.

### XLM-R (Fine-Tuned)

Fine-tunes XLM-RoBERTa using Hugging Face Trainer for multi-class love classification.

### VAD Regression

Uses ridge regression on `[CLS]` embeddings from the fine-tuned model to predict valence, arousal, and dominance.

## Results

The rule-based classifier performed well on romantic texts but poorly on others. Fine-tuned XLM-R achieved the best overall accuracy and F1 score. VAD regression produced low MAE scores, suggesting the model captures emotional gradients effectively.

## Conclusion

This project demonstrates that large multilingual models like XLM-RoBERTa can effectively classify types of love and predict affective attributes across diverse and noisy inputs. Fine-tuning was critical in helping the model adapt to domain-specific emotional language, outperforming zero-shot approaches.  
Challenges remain in figurative language, code-switching, and ambiguous phrasing. Future work could explore cross-modal sentiment, incorporating audio and visual data or real-time social media analysis.

## Notes for macOS/M1 Users

- Training uses Apple's MPS backend, so set `problem_type="single_label_classification"` to avoid MSE loss errors.
- `pin_memory=True` is safely ignored by PyTorch MPS — this warning can be disregarded.

## References

- Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)  
- Akbik et al. (2021). *Multilingual contextual word embeddings for sequence labeling*. [arXiv:1902.00193](https://arxiv.org/abs/1902.00193)  
- Mohammad et al. (2018). *Affect intensity and valence lexicons*. [arXiv:1805.00720](https://arxiv.org/abs/1805.00720)
- [Poetry Foundation](https://www.poetryfoundation.org/)  
- [Project Gutenberg](https://www.gutenberg.org/)  
- [OPUS](https://opus.nlpl.eu/)
