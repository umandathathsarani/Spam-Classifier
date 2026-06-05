import pandas as pd
import urllib.request
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

data_path = 'data/spam.tsv'
url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'

if not os.path.exists('data'):
    os.makedirs('data')

if not os.path.exists(data_path):
    urllib.request.urlretrieve(url, data_path)

df = pd.read_csv(data_path, sep='\t', header=None, names=['label', 'message'])

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

vectorizer = CountVectorizer(stop_words='english')
X_train_dtm = vectorizer.fit_transform(X_train)
X_test_dtm = vectorizer.transform(X_test)

nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)

y_pred_class = nb.predict(X_test_dtm)

print(f"Accuracy: {accuracy_score(y_test, y_pred_class):.4f}\n")
print(classification_report(y_test, y_pred_class))

if not os.path.exists('model'):
    os.makedirs('model')

joblib.dump(vectorizer, 'model/vectorizer.pkl')
joblib.dump(nb, 'model/classifier.pkl')