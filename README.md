

# 🤖 BERT Sentiment Analysis System

**An NLP and Deep Learning project that fine-tunes BERT (`bert-base-uncased`) using PyTorch and Hugging Face Transformers to classify social media text into Negative, Neutral, and Positive sentiments, with an interactive Streamlit application.**

> **Project Status: Prototype**
>
> This project demonstrates an end-to-end Transformer-based sentiment analysis workflow. The saved experiment used limited training and evaluation batches because of CPU execution constraints. Its results are preliminary and should not be interpreted as the model's full potential.

## 📌 Overview

The BERT Sentiment Analysis System is a Natural Language Processing and Deep Learning project designed to identify the sentiment expressed in short-form social media text.

The system uses the TweetEval Sentiment dataset from Hugging Face and fine-tunes a pretrained BERT model (`bert-base-uncased`) for three-class sentiment classification.

Using PyTorch and Hugging Face Transformers, the project implements text preprocessing, WordPiece tokenization, attention masking, model fine-tuning, and performance evaluation. It also includes an interactive prediction function and Streamlit application code for analyzing custom text and displaying sentiment confidence scores.

## 🎯 Project Objectives

The primary objectives of this project are to:

- Develop a Transformer-based sentiment classification system using pretrained BERT.
- Classify short social media text into Negative, Neutral, and Positive categories.
- Explore and preprocess real-world textual data using Natural Language Processing techniques.
- Understand BERT tokenization, attention masks, padding, and sequence truncation.
- Fine-tune a pretrained language model using PyTorch.
- Implement an optimization strategy using AdamW, learning rate scheduling, and gradient clipping.
- Evaluate model performance using accuracy, precision, recall, F1-score, and confusion matrices.
- Develop an interactive sentiment prediction interface using Streamlit.

## ✨ Key Features

- **Three-Class Sentiment Classification:** Identifies Negative, Neutral, and Positive sentiments.
- **Pretrained BERT Architecture:** Uses `bert-base-uncased` with a three-class classification head.
- **Hugging Face Integration:** Loads the TweetEval dataset, pretrained tokenizer, and BERT model.
- **Text Preprocessing:** Normalizes whitespace while preserving hashtags, mentions, and punctuation.
- **WordPiece Tokenization:** Converts text into token IDs using the BERT tokenizer.
- **Attention Masking:** Distinguishes meaningful tokens from padding tokens.
- **Custom PyTorch Training Loop:** Implements model training and validation using PyTorch.
- **Advanced Optimization:** Uses AdamW, linear learning rate warm-up, weight decay, and gradient clipping.
- **Model Evaluation:** Generates classification reports and confusion matrices.
- **Interactive Predictions:** Accepts custom text and returns predicted sentiment with confidence scores.
- **Streamlit Application:** Includes code for an interactive sentiment analysis web interface.
- **Model Saving:** Saves fine-tuned model weights and tokenizer files for reuse.

## 📊 Dataset

This project uses the **TweetEval Sentiment Dataset**, available through Hugging Face.

**Dataset source:** [cardiffnlp/tweet_eval](https://huggingface.co/datasets/cardiffnlp/tweet_eval)

The sentiment configuration contains short social media messages labeled according to their expressed sentiment.

### Dataset Summary

| Dataset Split | Original Samples |
|---|---:|
| Training | 45,615 |
| Validation | 2,000 |
| Testing | 12,284 |
| **Total** | **59,899** |

### Data Cleaning

- Checked all dataset splits for missing values.
- Found 26 duplicate records in the training dataset.
- Removed duplicate training records.
- Found no duplicates in validation or testing data.
- Retained 45,589 unique training samples.

### Sentiment Categories

| Label | Sentiment |
|---|---|
| 0 | Negative |
| 1 | Neutral |
| 2 | Positive |

**Dataset Observation:** Neutral sentiment is the dominant class. This class distribution creates a potential classification imbalance that makes class-wise evaluation particularly important.

## ⚙️ Methodology

The project follows a structured Transformer-based NLP workflow.

### 1. Dataset Loading

- Loads TweetEval using Hugging Face Datasets.
- Retrieves training, validation, and testing splits.
- Converts dataset splits into Pandas DataFrames.
- Checks dataset structure and label distribution.

### 2. Exploratory Data Analysis

- Checks missing values and duplicate records.
- Removes duplicate training entries.
- Examines sentiment class distribution.
- Visualizes class frequencies using Matplotlib and Seaborn.

### 3. Text Preprocessing

A custom `clean_tweet_text()` function is implemented to:

- Normalize multiple spaces.
- Remove unnecessary whitespace.
- Trim leading and trailing spaces.
- Preserve hashtags, mentions, and punctuation because they can provide useful sentiment signals.

### 4. BERT Tokenization

Uses the pretrained `bert-base-uncased` tokenizer to convert text into numerical representations.

The tokenization process includes:

- WordPiece subword tokenization.
- Special tokens such as `[CLS]` and `[SEP]`.
- Input IDs.
- Attention masks.
- Padding and truncation.
- Maximum sequence length of 128 tokens.

### 5. Model Architecture

The project uses `BertForSequenceClassification` with a three-class classification head.

| Component | Configuration |
|---|---|
| Pretrained Model | `bert-base-uncased` |
| Transformer Layers | 12 |
| Hidden Size | 768 |
| Attention Heads | 12 |
| Classification Classes | 3 |
| Loss Function | Cross-Entropy Loss |

The BERT encoder generates contextual text representations, while the classification head predicts the sentiment category.

### 6. Model Training

A custom PyTorch `Dataset` class and `DataLoader` instances are implemented for training and validation.

The model is fine-tuned using the following configuration:

| Hyperparameter | Value |
|---|---:|
| Learning Rate | 0.00002 (2e-5) |
| Batch Size | 32 |
| Epochs | 3 |
| Weight Decay | 0.01 |
| Maximum Sequence Length | 128 |
| Optimizer | AdamW |
| Learning Rate Scheduler | Linear Warm-up |
| Gradient Clipping | 1.0 |
| Loss Function | Cross-Entropy |

**Prototype execution:** Because the saved run used CPU execution, training was limited to 50 batches per epoch, approximately 1,600 tweets per epoch. Validation was limited to 15 batches per epoch.

### 7. Model Evaluation

Evaluates the trained model using:

- Accuracy
- Precision
- Recall
- F1-score
- Classification reports
- Confusion matrices

The saved experiment evaluates a limited sample of the test dataset rather than the complete testing split.

### 8. Interactive Sentiment Prediction

A custom `predict_sentiment()` function is developed to:

- Accept user-provided text.
- Tokenize the input using BERT.
- Generate model predictions.
- Calculate class probability scores using Softmax.
- Return the predicted sentiment and confidence score.

### 9. Model Saving

The notebook saves:

- Fine-tuned model weights as `bert_sentiment_model.pt`.
- Tokenizer files inside the `saved_tokenizer/` directory.

These files can be reused for inference without repeating the entire training process.

### 10. Streamlit Application

The notebook also generates an interactive Streamlit application named `app.py`.

The application includes:

- Text input for tweets or sentences.
- A sentiment prediction button.
- Predicted sentiment labels.
- Confidence score display.
- Individual probability bars for Negative, Neutral, and Positive classes.

## 📈 Model Performance & Results

The saved experiment demonstrated improving validation accuracy across three training epochs.

### Validation Performance

| Epoch | Training Accuracy | Validation Accuracy |
|---|---:|---:|
| 1 | 34.88% | 43.96% |
| 2 | 46.81% | 50.00% |
| 3 | 56.13% | 56.46% |

### Test Evaluation

The model was evaluated on a sample of 960 tweets from the test dataset.

<badge color="warning">Prototype result</badge>

<title size="2xl" color="default" tabularNums>55%</title>
<text color="secondary" size="xs">Overall test accuracy on the evaluated 960-tweet sample</text>

### Classification Report

| Sentiment | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Negative | 0.00 | 0.00 | 0.00 | 293 |
| Neutral | 0.54 | 0.93 | 0.68 | 479 |
| Positive | 0.65 | 0.46 | 0.54 | 188 |
| **Macro Average** | **0.40** | **0.46** | **0.41** | **960** |
| **Weighted Average** | **0.40** | **0.55** | **0.45** | **960** |

### Key Findings

- Validation accuracy increased from 43.96% to 56.46% across the three epochs.
- The model achieved 55% accuracy on the evaluated test sample.
- Neutral sentiment received the highest recall at 0.93.
- The model did not correctly identify any Negative samples.
- The model showed a tendency to predict Neutral sentiment.
- Training loss continued to decrease, indicating that the model had not fully converged during the saved experiment.

**Important:** These results represent a limited prototype experiment, not a full evaluation of BERT on the TweetEval dataset.

## ⚠️ Limitations

The current implementation has several limitations:

- Training used only 50 batches per epoch on CPU rather than the complete training dataset.
- Testing covered only 960 tweets instead of all 12,284 test samples.
- The model failed to identify the Negative class in the evaluated sample.
- The class distribution may have contributed to prediction bias.
- No traditional machine learning baseline, such as TF-IDF with Logistic Regression, was included.
- The Streamlit application requires the saved fine-tuned weights and tokenizer to provide meaningful trained-model predictions.

These limitations provide opportunities for further experimentation and improvement.

## 🔮 Future Improvements

- Fine-tune BERT on the complete training dataset using GPU acceleration.
- Evaluate the model on the full test dataset.
- Improve performance across all three sentiment classes.
- Investigate class imbalance using class weights or other suitable techniques.
- Report macro-F1 and weighted-F1 after full evaluation.
- Compare BERT against a TF-IDF and Logistic Regression baseline.
- Experiment with different learning rates, batch sizes, and training epochs.
- Improve model saving and loading for reliable deployment.
- Package the Streamlit application with the required model files and dependencies.

## 🛠️ Technologies & Libraries

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| NLP Architecture | BERT |
| Pretrained Model | `bert-base-uncased` |
| NLP Framework | Hugging Face Transformers |
| Dataset Management | Hugging Face Datasets |
| Data Processing | Pandas, NumPy |
| Machine Learning Evaluation | Scikit-learn |
| Data Visualization | Matplotlib, Seaborn |
| Web Application | Streamlit |
| Development Environment | Google Colab / Jupyter Notebook |

## 📁 Project Structure

The notebook generates additional application and model files when the relevant cells are executed.

```text
Sentiment-Analysis/
│
├── Sentiment_Analysis.ipynb
├── sentiment_analysis.py
├── app.py
├── requirements.txt
│
├── bert_sentiment_model.pt
│
└── saved_tokenizer/
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    ├── vocab.txt
    └── tokenizer files
```

**Note:** `app.py`, `requirements.txt`, the saved model weights, and tokenizer directory are generated or saved by the notebook. They must be included in the repository if you want others to run the application directly. The Python export is included in this structure as a repository file, as described in the project documentation.

## 🚀 Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/alishaarif500/Sentiment-Analysis.git
```

### 2. Navigate to the Project Directory

```bash
cd Sentiment-Analysis
```

### 3. Install Dependencies

```bash
pip install torch transformers datasets scikit-learn pandas numpy matplotlib seaborn streamlit jupyter
```

### 4. Run the Notebook

```bash
jupyter notebook Sentiment_Analysis.ipynb
```

Alternatively, upload the notebook to Google Colab.

**Recommended:** Use a GPU runtime for faster BERT fine-tuning.

Execute the notebook cells in order to:

- Load the TweetEval dataset.
- Preprocess and tokenize the text.
- Configure and fine-tune BERT.
- Evaluate the model.
- Generate the prediction function.
- Save model weights and tokenizer files.
- Generate the Streamlit application and requirements file.

### 5. Launch the Streamlit Application

After generating the application and saving the trained model and tokenizer files, execute:

```bash
streamlit run app.py
```

The application will be available locally at:

```text
http://localhost:8501
```

**Important:** The Streamlit application expects `bert_sentiment_model.pt` and `saved_tokenizer/` to be available. Without the fine-tuned weights, its fallback loads the base BERT architecture with a newly initialized classification head, which should not be treated as a trained sentiment classifier.

## 🎓 Learning Outcomes

Through this project, I gained practical experience in:

- Understanding Transformer-based Natural Language Processing.
- Working with pretrained BERT architectures.
- Loading and exploring datasets using Hugging Face Datasets.
- Performing text preprocessing and data cleaning.
- Understanding WordPiece tokenization and attention masks.
- Building custom PyTorch Dataset and DataLoader pipelines.
- Fine-tuning pretrained language models for sequence classification.
- Implementing AdamW optimization and learning rate scheduling.
- Applying gradient clipping during model training.
- Evaluating multi-class classification performance.
- Analyzing class-wise results and confusion matrices.
- Developing interactive text prediction functionality.
- Creating a Streamlit interface for machine learning inference.

## 👩‍💻 Developer

**Alisha Arif**

- **GitHub:** [github.com/alishaarif500](https://github.com/alishaarif500)
- **LinkedIn:** [linkedin.com/in/alisha-arif-22aa42386](https://linkedin.com/in/alisha-arif-22aa42386)
- **Email:** [alishaarif500@gmail.com](mailto:alishaarif500@gmail.com)

---

⭐ If you find this project useful, consider giving the repository a star!

