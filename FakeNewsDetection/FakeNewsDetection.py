# Essential Libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Load Dataset
dataset = pd.read_csv('news.csv')
print(dataset.head())

# Check dataset shape
print(dataset.shape)

# Get Labels
labels = dataset.label
print(labels.head())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset['text'], labels, test_size = 0.2, random_state = 7)

# Initialize the TfidVectorizer
tfidVectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.7)

# Fit and transform train set, transform test set
tfid_tain = tfidVectorizer.fit_transform(X_train)
tfid_test = tfidVectorizer.transform(X_test)

# Initialize a PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter = 50)
pac.fit(tfid_tain, y_train)

# Predict on the test set and calculate accuracy
y_pred = pac.predict(tfid_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score*100,2)}%')

# Build confusion matrix
print(confusion_matrix(y_test,y_pred, labels=['FAKE','REAL']))