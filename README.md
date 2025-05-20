# Sentiment Analysis for Automotive Reviews

This project is an AI-powered sentiment analysis web application designed to predict the sentiment of automotive reviews using various machine learning models. The web app is built using Flask, allowing users to input their reviews and get real-time predictions on the sentiment.

## Features

- **Machine Learning Models:** Naive Bayes, Logistic Regression, Random Forest, SVM, and Ensemble Voting Classifier.
- **Text Preprocessing:** Includes tokenization, stopword removal, and lemmatization.
- **Vectorization:** TF-IDF vectorization of user input text.
- **Model Persistence:** Pickled models and vectorizer for deployment.
- **Web Interface:** Built using Flask for sentiment prediction based on user input.

## Requirements

- Python 3.x
- Flask
- scikit-learn
- pandas
- numpy
- nltk (for text preprocessing)
- pickle (for saving models)
  
Install required packages with:
```bash
pip install -r requirements.txt
```
## Dataset

You can use any CSV file for training the model as long as it includes columns for cleaned reviews and the target labels (detailed emotions). Replace the dataset with your own and ensure that you correctly define `X` and `y` in the code:

- **X:** Your column for text reviews (e.g., `clean_review`).
- **y:** Your column for emotion labels (e.g., `detailed_emotion`).

Example:

```python
df = pd.read_csv('your_dataset.csv')
X = df['your_review_column']
y = df['your_emotion_label_column']
```
## Running the App

1. **Train the Models:**
    - Run the Python script to train models using your dataset.
    - The models and vectorizer will be saved using pickle (`ensemble_model.pkl` and `vectorizer_final.pkl`).

2. **Start Flask App:**
    - Start the Flask server to use the sentiment analysis interface.

    ```bash
    python app.py
    ```

3. **Access the Web App:**
    - Open your browser and navigate to `http://127.0.0.1:5000/` to use the sentiment prediction feature.

## How to Use

- Enter your automotive review in the text box.
- Click **Predict** to get the sentiment result.
- The **Clear** button resets the prediction.

## Example of Usage

1. **Input:** "This car has amazing fuel efficiency!"
2. **Output:** "Happy" (predicted emotion)

## Model Overview

The following models are used for sentiment prediction:

- **Naive Bayes (MultinomialNB)**
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Random Forest Classifier**
- **Ensemble Voting Classifier** (combining SVM and Logistic Regression)

The ensemble model has been pickled for use in the Flask app to provide accurate predictions based on user input.

## Saving and Loading Models

- The trained ensemble model and vectorizer are saved in `.pkl` files.
- To use the same setup, ensure your files (`ensemble_model.pkl` and `vectorizer_final.pkl`) are in the root directory of your Flask app.
## Sample output
![Screenshot_10-10-2024_15138_](https://github.com/user-attachments/assets/026c3a1f-42ef-4c02-86f2-4256281337b5)
![Screenshot 2024-04-21 194439](https://github.com/user-attachments/assets/b53a9ed0-7749-4598-a3df-c86910e579a5)
![Screenshot 2024-04-21 194459](https://github.com/user-attachments/assets/9de8a6c2-1707-442c-9fd7-d45a19af0d17)
![Screenshot 2024-04-21 194657](https://github.com/user-attachments/assets/604d0280-11aa-4cb5-b63b-513d1028cc04)
![Screenshot 2024-04-21 194709](https://github.com/user-attachments/assets/3a7c460b-53ac-4f16-9787-e998f17be5b2)

