# Automatic Tagging of Coding-Related Questions

This repository contains the code and resources to train a machine learning model for tagging coding-related questions with predefined categories. It’s designed to improve the organization of questions on coding forums, Q&A platforms, and support websites.

---

## Overview

The project leverages a **Support Vector Machine (SVM)** classifier combined with **TF-IDF vectorization** to classify coding-related questions into relevant tags.

---

## Dataset

- Source: Coding-related questions collected from GitHub.
- Structure: Questions paired with their corresponding tags.

---

## Model Architecture

1. **TF-IDF Vectorization**: Converts text data into numerical features.
2. **SVM Classifier**: A linear model trained on the vectorized data for tag prediction.

---

## Training Pipeline

1. **Data Preprocessing**:
   - Noise removal
   - Tokenization
   - Stopword removal
   - Lemmatization

2. **Feature Extraction**:
   - TF-IDF vectorization for numerical representation.

3. **Model Training**:
   - Linear SVM classifier trained on preprocessed data.

4. **Evaluation**:
   - Performance measured on a validation set using metrics like accuracy, precision, and recall.

---
