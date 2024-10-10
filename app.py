from flask import Flask, render_template, request 
import pickle 
from sklearn.feature_extraction.text import TfidfVectorizer 
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer 
import string 

app = Flask(__name__) 
 
# Load the vectorizer and ensemble model from pickle files 
with open('vectorizer_final.pkl', 'rb') as file: 
    vectorizer = pickle.load(file) 
 
with open('ensemble_model.pkl', 'rb') as file: 
    model = pickle.load(file) 
 
# Define the preprocess_text function 
def preprocess_text(text): 
    # Tokenization 
    tokens = word_tokenize(text) 
     
    # Lowercasing 
    tokens = [token.lower() for token in tokens] 
     
    # Removing punctuation 
    tokens = [token for token in tokens if token not in string.punctuation] 
     
    # Removing stopwords 
    stop_words = set(stopwords.words('english')) 
    tokens = [token for token in tokens if token not in stop_words] 
     
    # Lemmatization 
    lemmatizer = WordNetLemmatizer() 
    tokens = [lemmatizer.lemmatize(token) for token in tokens] 
     
    # Join tokens back into text 
    preprocessed_text = ' '.join(tokens) 
    print(preprocessed_text) 
    return preprocessed_text 
 
@app.route('/') 
def index(): 
    return render_template('index.html') 
 
@app.route('/predict', methods=['POST']) 
def predict(): 
    if request.method == 'POST': 
        text = request.form['text'] 
         
        # Preprocess the input text 
        preprocessed_text = preprocess_text(text) 
         
        # Transform the preprocessed text using the loaded vectorizer 
        text_vectorized = vectorizer.transform([preprocessed_text]) 
         
        # Make prediction using the loaded ensemble model 

        prediction = model.predict(text_vectorized) 
         
        return render_template('index.html', prediction=prediction[0]) 
 
if __name__ == '__main__': 
    app.run(debug=True) 