# Sentiment-Analysis
##  BERT Sentiment Analysis System & Interactive Web Dashboard
# Sentiment Analysis with BERT (TweetEval)

A prototype that fine-tunes `bert-base-uncased` with PyTorch and Hugging Face Transformers to classify short social-media text as negative, neutral or positive.

> **Status: prototype.** The saved run used a heavily reduced amount of training data and compute, and its results (below) are weak. They do not show what BERT can achieve on this dataset.

## Overview

The notebook loads the TweetEval sentiment dataset, cleans it, tokenises text with the BERT tokenizer, fine-tunes the model in a custom PyTorch training loop (AdamW optimizer with linear warm-up schedule and gradient clipping), and evaluates it with a classification report and confusion matrix.

## Dataset

- **Source:** TweetEval (sentiment), `cardiffnlp/tweet_eval` on Hugging Face.
- **Splits:** 45,615 training tweets (45,589 after removing 26 duplicates), 2,000 validation and 12,284 test tweets.
- **Labels:** 0 = Negative, 1 = Neutral, 2 = Positive.

## Methodology

- **Model:** `bert-base-uncased` with a 3-class classification head.
- **Settings:** learning rate 2e-5, batch size 32, 3 epochs, weight decay 0.01, maximum sequence length 128.
- **Saved run:** each epoch processed 50 batches of 32 tweets (about 1,600 tweets), a small fraction of the training set.

## Results

Validation accuracy over the three epochs of the saved run: 0.440, 0.500, 0.565. Training loss was still falling, which suggests the model was under-trained.

Test results on 960 tweets (overall accuracy **0.55**):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative | 0.00 | 0.00 | 0.00 | 293 |
| Neutral | 0.54 | 0.93 | 0.68 | 479 |
| Positive | 0.65 | 0.46 | 0.54 | 188 |

The model never predicted the Negative class and mostly favoured Neutral.

## Limitations

- Training used only a small portion of the data for a few steps, so the model is clearly under-trained.
- Evaluation was on a 960-tweet sample, not the full 12,284-tweet test set.
- No comparison with a simple baseline (for example TF-IDF with Logistic Regression) is included.
- The repository does not include a web app.

## Future Improvements

- Retrain on the full training set with a GPU runtime and report results on the full test set.
- Report macro-F1 and compare with a TF-IDF baseline.
- Handle class imbalance and test other settings (learning rate, epochs).
- Add the saved model or a small demo app.

## Project Files

| File | Description |
|---|---|
| `Sentiment_Analysis.ipynb` | Main notebook: data loading, tokenisation, fine-tuning, evaluation |
| `sentiment_analysis.py` | Python export of the notebook |

## How to Run

The notebook was built in Google Colab and downloads the dataset and the pre-trained model from Hugging Face. A GPU runtime is strongly recommended.

**Google Colab (recommended):** upload the notebook, select a GPU runtime and run all cells.

**Locally:**

```bash
pip install torch transformers datasets scikit-learn pandas matplotlib seaborn jupyter
jupyter notebook Sentiment_Analysis.ipynb
```

## Technologies

Python · PyTorch · Hugging Face Transformers and Datasets · scikit-learn · Matplotlib · Seaborn · Google Colab

## Author

**Alisha Arif** · [GitHub](https://github.com/alishaarif500) · [LinkedIn](https://linkedin.com/in/alisha-arif-22aa42386/)
This repository hosts a production-grade, state-of-the-art NLP pipeline built with PyTorch and Hugging Face Transformers. By fine-tuning a pre-trained `bert-base-uncased` encoder model, the system classifies short-form social texts (tweets, feedback, reviews) into three semantic classes: **Positive**, **Neutral**, and **Negative**.

###  Key Features:
- **Advanced Fine-Tuning:** Custom PyTorch training loops featuring the AdamW optimizer, linear learning rate warm-ups, and robust gradient clipping.
- **Modern NLP Tokenization:** Implements Hugging Face's subword tokenization with attention masking, padding, and sequence truncation optimized for sequence classification.
- **Real-Time Interactive App:** Includes a containerizable Streamlit web application (`app.py`) with predictive score confidence distribution charts.
- **Comprehensive Evaluation:** Built-in scripts for reporting test-set metrics (Accuracy, Precision, Recall, and F1-Scores) and plotting confusion matrices.
