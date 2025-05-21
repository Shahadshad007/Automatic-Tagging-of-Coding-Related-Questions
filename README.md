# Automatic Tagging of Coding-Related Questions

This repository contains the code and resources to train a machine learning model for tagging coding-related questions with predefined categories. It’s designed to improve the organization of questions on coding forums, Q&A platforms, and support websites.

---

## Overview

The project leverages multiple machine learning models, including **Support Vector Machine (SVM)**, **K-Nearest Neighbors (KNN)**, **Decision Tree**, **Random Forest**, and **Naive Bayes**, combined with **TF-IDF Vectorization** and **Count Vectorization** to classify coding-related questions into relevant tags. The best-performing model (SVM with TF-IDF) achieves an accuracy of 57%.

---

## Dataset

- **Source**: Coding-related questions collected from GitHub.
- **Structure**: The dataset contains 22,164 questions with columns: `Title`, `Body` (dropped during preprocessing), and `Tag` (e.g., `javascript`, `java`, `android`).
- **Tag Distribution**:
  - `javascript`: 3,272
  - `java`: 3,115
  - `c#`: 2,723
  - `php`: 2,680
  - `android`: 2,498
  - `jquery`: 2,099
  - `python`: 1,726
  - `html`: 1,527
  - `c++`: 1,298
  - `ios`: 1,226

---

## Model Architecture

1. **TF-IDF Vectorization**: Converts preprocessed question titles into numerical features using `TfidfVectorizer`.
2. **Count Vectorization**: Converts preprocessed question titles into numerical features using `CountVectorizer` for comparison.
3. **Models**:
   - **Support Vector Machine (SVM)**: Best performer with 57% accuracy using TF-IDF.
   - **K-Nearest Neighbors (KNN)**: Distance-based classifier.
   - **Decision Tree**: Tree-based classifier.
   - **Random Forest**: Ensemble of decision trees.
   - **Naive Bayes**: Probabilistic classifier (`MultinomialNB`).

---

## Training Pipeline

1. **Data Preprocessing**:
   - Dropped the `Body` column, used only the `Title` column.
   - Removed HTML tags using regex.
   - Removed special characters and digits, keeping only alphabetical characters.
   - Tokenization using NLTK’s `word_tokenize`.
   - Removed stopwords using NLTK’s English stopwords list.
   - Lemmatization using NLTK’s `WordNetLemmatizer`.

2. **Feature Extraction**:
   - Applied `TfidfVectorizer` to create `x_train_tfidf` and `x_test_tfidf`.
   - Applied `CountVectorizer` to create `x_train_count` and `x_test_count`.

3. **Data Splitting**:
   - Split dataset into 70% training and 30% testing sets (`random_state=42`).

4. **Model Training**:
   - Encoded `Tag` column using `LabelEncoder` to create `Tag_encoded`.
   - Trained models (KNN, Decision Tree, Random Forest, SVM, Naive Bayes) on both TF-IDF and Count Vectorized data.

5. **Evaluation**:
   - **TF-IDF Vectorized Data**:
     - KNN: 40% accuracy
     - Decision Tree: 48% accuracy
     - Random Forest: 56% accuracy
     - SVM: 57% accuracy (highest)
     - Naive Bayes: 50% accuracy
   - **Count Vectorized Data**:
     - KNN: 40% accuracy
     - Decision Tree: 49% accuracy
     - Random Forest: 55% accuracy
     - SVM: 55% accuracy
     - Naive Bayes: 56% accuracy
   - Visualized accuracies using bar plots.

6. **Model Saving**:
   - Saved the best model (SVM with TF-IDF) as `svm_model_tfidf.joblib`.
   - Saved the TF-IDF vectorizer as `tfidf_vectorizer.joblib`.

---
