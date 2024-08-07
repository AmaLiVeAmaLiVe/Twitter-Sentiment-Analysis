# Twitter sentiment analysis.

The twitter-sentiment-analysis.ipynb file is a Jupyter Notebook that contains a data analysis with a several machine learning techniques. In this project, the data was sourced from <a href="https://www.kaggle.com/datasets/kazanova/sentiment140">here</a> and <a href="https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis">here </a>. The model classifies tweets into positive and negative sentiments. 

This project aims to provide an end-to-end solution for sentiment analysis on Twitter data. It includes data preprocessing, model training, evaluation, and testing. The best-performing model is saved for future predictions.


## Features

- **Data Preprocessing:** Functions to clean and preprocess Twitter data, including removing stopwords, punctuation, URLs, and numbers.
- **Exploratory Data Analysis:** Jupyter notebook for visualizing and understanding the data.
- **Model Training:** Scripts and notebooks for training various machine learning models and selecting the best one.
- **Model Evaluation:** Comprehensive evaluation metrics including accuracy, F1 score, and ROC AUC score.
- **Sentiment Analysis:** Script to analyze new tweets and predict their sentiment.
- **Model Serialization:** Saving the best model and vectorizer for future use.
- **Unit Testing:** Tests to ensure the reliability of preprocessing functions.


## Installation

1. Clone the repository:

   git clone https://github.com/yourusername/twitter-sentiment-analysis.git
   
   cd twitter-sentiment-analysis

2. Install the required packages:

   pip install -r requirements.txt

## Usage

1. Twitter Sentiment analysis:
   
   jupyter notebook src/twitter_sentiment_analysis.ipynb

2. Testing the model:
   
   jupyter notebook src/testing_model.ipynb

3. Tests:
   
   python tests/test_preprocessing.py
