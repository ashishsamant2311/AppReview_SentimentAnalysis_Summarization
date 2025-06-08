# Google Application Reviews - Sentiment Analysis using RoBERTa and Summarization using bart-large-cnn

This repository provides an end-to-end pipeline for analyzing app reviews using state-of-the-art transformer models. The goal is to classify sentiments and generate concise summaries of reviews, leveraging Hugging Face's powerful models and tools.

---

## Table of Contents

1. [Purpose](#purpose)
2. [Use Case](#use-case)
3. [Datasets](#datasets)
4. [Models Used](#models-used)
5. [Implementation](#implementation)
6. [Acknowledgements](#acknowledgements)

---

## Purpose

The primary purpose of this project is to extract insights from user reviews. By identifying sentiment polarity and summarizing reviews, the pipeline provides actionable information to improve customer experience and product quality.

---

## Use Case

### Who Can Use This?
- **App Developers:** Understand user feedback to enhance app features.
- **Product Managers:** Identify key themes in customer reviews to guide improvements.
- **Data Scientists:** Explore transformer-based sentiment analysis and summarization techniques.

### Why Is This Useful?
- **Sentiment Analysis:** Highlight strengths and weaknesses based on positive and negative reviews.
- **Summarization:** Quickly derive insights from large volumes of reviews.

---

## Datasets

### `sealuzh/app_reviews`
This dataset contains app reviews and their corresponding ratings, sourced from Hugging Face Datasets.
- **Features:**
  - **Review Text:** User-provided feedback.
  - **Rating:** Numerical rating (used for creating sentiment labels).
- **Applications:** Sentiment classification and summarization.

For more details, visit [sealuzh/app_reviews](https://huggingface.co/datasets/sealuzh/app_reviews).

---

## Models Used

### 1. **RoBERTa for Sentiment Analysis**
- **Purpose:** Classifies reviews into positive, negative, or neutral sentiments.
- **Why RoBERTa?**
  - Pre-trained on a large corpus of text for better contextual understanding.
  - Fine-tuned for sentiment classification to maximize accuracy.

### 2. **BART for Summarization**
- **Purpose:** Generates concise summaries of reviews.
- **Why BART?**
  - A denoising autoencoder for sequence-to-sequence tasks.
  - Excels at abstractive summarization.

---

## Implementation

1. **Data Preparation:** Load the `app_reviews` dataset and label data based on ratings to create supervised learning inputs for sentiment analysis.
2. **Sentiment Analysis:** Fine-tune the RoBERTa model to classify sentiments and rank reviews by sentiment probability.
3. **Summarization:** Use BART to create concise summaries of the reviews, highlighting key aspects.
4. **Optimization:** Utilize `bitsandbytes` for quantization to enable efficient model deployment on resource-constrained systems.

---

## Acknowledgements

- **Hugging Face:** For providing state-of-the-art transformer models and datasets.
- **Sealuzh Research Group:** For the comprehensive `app_reviews` dataset.
- **PyTorch Community:** For enabling efficient model training and deployment.
- **OpenAI Community:** For inspiring work in NLP and transformers.
