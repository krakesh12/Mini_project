from flask import Flask, request, render_template
import joblib
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from nltk.tokenize import word_tokenize

app = Flask(__name__)

loaded_model = joblib.load('sentiment_model.pkl')
vectoriser = joblib.load('vector.pkl')

# Function to preprocess text
def preprocess_text(text):
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # Convert to lowercase and tokenize
    words = word_tokenize(text.lower())

    # Remove stopwords
    words = [word for word in words if word not in stopwords.words('english')]

    # Join the cleaned words back into a sentence
    cleaned_text = ' '.join(words)

    return cleaned_text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        cleaned_text = preprocess_text(text)
        vectorized_text = vectoriser.transform([cleaned_text])
        prediction = loaded_model.predict(vectorized_text)
        sentiment = "Positive" if prediction[0] == 0 else "Negative"
        return render_template('index.html', prediction_text=f'{sentiment}')

if __name__ == '__main__':
    app.run(debug=True)
