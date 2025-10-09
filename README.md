# Product Review Sentiment Analysis

## Overview

This project demonstrates how to build and deploy a sentiment analysis model from scratch to classify product reviews as Positive or Negative.

The model is trained using Word2Vec embeddings and an LSTM neural network, capturing the semantic meaning of words for better understanding of customer opinions. The project also includes an interactive Streamlit web application for real-time sentiment prediction.

This project was created as part of my learning journey in Machine Learning and NLP, focusing on building end-to-end ML pipelines — from raw data preprocessing to model training, evaluation, and deployment.

------------------------------------------------------------------------

## How to Run Locally

-  Step 1: Clone the Repository

    ``` bash
    git clone https://github.com/yourusername/product-review-sentiment-analysis.git
    cd product-review-sentiment-analysis

    ```

-  Step 2: Install Dependencies

    If using uv:

    ``` bash
    uv sync

    ```
    Or using pip:

    ``` bash
    pip install -r requirements.txt

    ```

-  Step 3: Run the Streamlit App
    ``` bash
    streamlit run main.py
    # For uv users:
    uv run streamlit run main.py
    ```

-  Step 4: Test the App

    Once running, open the local Streamlit URL (e.g., http://localhost:8501) and enter a review like:

    “The quality of this product is amazing!”
    → Output: Positive

    “Terrible experience, will not buy again.”
    → Output: Negative

------------------------------------------------------------------------

## Project Workflow
1. Data Preparation

    - Loaded and cleaned raw product reviews (reviews.csv).

    - Removed noise, punctuation, and stopwords.

    - Tokenized and normalized text using NLTK.

2. Feature Engineering

    - Trained a Word2Vec model to create word embeddings capturing semantic meaning.

    - Transformed reviews into padded vector sequences for uniform input length.

3. Model Building

    - Designed an LSTM-based neural network to learn contextual relationships between words.

    - Used dropout and regularization to prevent overfitting.

    - Optimized using binary cross-entropy loss and Adam optimizer.

4. Model Training & Evaluation

    - Trained the LSTM model on labeled product review data.

    - Achieved high accuracy in differentiating positive vs negative sentiments.

    - Addressed class imbalance using weighted loss and validation monitoring.

5. Deployment with Streamlit

    - Integrated the trained model and embeddings into a Streamlit app.

    - Users can input any product review and instantly see predicted sentiment.

------------------------------------------------------------------------

## Key Learnings

- Understanding text preprocessing and tokenization techniques.

- Building Word2Vec embeddings and integrating them with neural models.

- Implementing an LSTM architecture for text classification.

- Handling imbalanced datasets and model overfitting.

- Deploying a trained ML model using Streamlit for real-time inference.

------------------------------------------------------------------------

