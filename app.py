import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt')
nltk.download('stopwords')
import joblib
def preprocess_text(text):
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # Convert to lowercase and tokenize
    words = nltk.word_tokenize(text.lower())

    # Remove stopwords
    words = [word for word in words if word not in stopwords.words('english')]

    # Join the cleaned words back into a sentence
    cleaned_text = ' '.join(words)

    return cleaned_text

import pandas as pd
data=pd.read_csv('Tweets.csv', encoding='ISO-8859-1')
dt=data
dt.dropna(subset=['text'], inplace=True)
dt.drop('selected_text', inplace=True, axis=1)
dt.drop('sentiment', inplace=True, axis=1)
cleaned_data = [preprocess_text(text) for text in dt['text']]

dt['processed_tweets']=cleaned_data
print(dt.head())
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
dt['compound'] = [analyzer.polarity_scores(x)['compound'] for x in dt['processed_tweets']]
dt['neg'] = [analyzer.polarity_scores(x)['neg'] for x in dt['processed_tweets']]
dt['neu'] = [analyzer.polarity_scores(x)['neu'] for x in dt['processed_tweets']]
dt['pos'] = [analyzer.polarity_scores(x)['pos'] for x in dt['processed_tweets']]
dt['comp_score'] = dt['compound'].apply(lambda c: 0 if c >=0 else 1)
pd.DataFrame(dt)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dt['processed_tweets'],dt['comp_score'],test_size=0.3,random_state=42)
print('Number of rows in the total set: {}'.format(dt.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
vector = CountVectorizer(stop_words = 'english', lowercase = True)
training_data = vector.fit_transform(X_train)
testing_data = vector.transform(X_test)
# vectoriser = TfidfVectorizer()
# vectoriser.fit(X_train)
# training_data = vectoriser.transform(X_train)
# testing_data= vectoriser.transform(X_test)

from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score
svc=SVC(kernel='linear', C=1.0, random_state=42)
svc.fit(training_data,y_train)
y_pred=svc.predict(testing_data)
accuracy=accuracy_score(y_pred,y_test)
classification_rep = classification_report(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)

# Save the trained model to a .pkl file
joblib.dump(svc, 'sentiment_model.pkl')

# Save the vectorizer to a .pkl file
joblib.dump(vector, 'vector.pkl')