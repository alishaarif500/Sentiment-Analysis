# Sentiment-Analysis
##  BERT Sentiment Analysis System & Interactive Web Dashboard

This repository hosts a production-grade, state-of-the-art NLP pipeline built with PyTorch and Hugging Face Transformers. By fine-tuning a pre-trained `bert-base-uncased` encoder model, the system classifies short-form social texts (tweets, feedback, reviews) into three semantic classes: **Positive**, **Neutral**, and **Negative**.

###  Key Features:
- **Advanced Fine-Tuning:** Custom PyTorch training loops featuring the AdamW optimizer, linear learning rate warm-ups, and robust gradient clipping.
- **Modern NLP Tokenization:** Implements Hugging Face's subword tokenization with attention masking, padding, and sequence truncation optimized for sequence classification.
- **Real-Time Interactive App:** Includes a containerizable Streamlit web application (`app.py`) with predictive score confidence distribution charts.
- **Comprehensive Evaluation:** Built-in scripts for reporting test-set metrics (Accuracy, Precision, Recall, and F1-Scores) and plotting confusion matrices.
