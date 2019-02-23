# Text_classification.

## Overview
The repository implements the common algorithms for multi-class text classification:
- Word or char level representation, chi-square+tfidf, word2vec, glove, fasttext or the concated one, elmo, bert
- Model: CNN, BiLSTM, Self-attention,C-LSTM, RCNN, Capsule, HAN, SVM, XGBoost

## Dependencies
- Python 3.6
- Tensorflow 1.12.0

## Usage
python run_classifier.py
- for new data, set the new_data=True, the model will generate the tf_record and usage the model_params.py
- for existed data, set new_data=False, the model will run from the saved tf_record and config.json

## Purpose
The classification is used to clarify the damaged part and damage type from vehicles comments

## Evaluation
Due to we have too many categories of labels (ca. 500 class for 100,000 examples), and they are not equally important, so we don’t use Macro- evaluation. And the Micro- precision/recall/F1 is the same for multi-label classification. So we check the accuracy and weighted F1

## Ignored issue
- Sometimes in one sample, more than one label are valid
- Some labels have hierarchy relationship
