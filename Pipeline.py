import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load the data
with open('twitter-datasets/train_pos_full.txt', 'r') as f:
    positive_tweets = f.readlines()
with open('twitter-datasets/train_neg_full.txt', 'r') as f:
    negative_tweets = f.readlines()

# Create DataFrame
data = pd.DataFrame({
    'text': positive_tweets + negative_tweets,
    'label': [1] * len(positive_tweets) + [0] * len(negative_tweets)
})

def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove mentions, hashtags, and special characters
    text = re.sub(r'[@#]\w+|[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

# Preprocess text data
data['text'] = data['text'].apply(preprocess_text)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42
)

# Feature extraction: TF-IDF
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model: Logistic Regression
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train_tfidf, y_train)

# Evaluate on validation set
y_pred = model.predict(X_test_tfidf)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Prepare for submission
with open('twitter-datasets/test_data.txt', 'r') as f:
    test_tweets = f.readlines()

# Preprocess test data
test_tweets = [preprocess_text(tweet) for tweet in test_tweets]

# Transform test data using TF-IDF
test_tfidf = vectorizer.transform(test_tweets)

# Make predictions
predictions = model.predict(test_tfidf)

# Convert predictions to -1 and 1 format
predictions = [1 if pred == 1 else -1 for pred in predictions]

# Create a submission DataFrame
submission = pd.DataFrame({
    'Id': range(1, len(predictions) + 1),
    'Prediction': predictions
})

# Save to CSV
submission.to_csv('submissionNEWp.csv', index=False)

print("Submission file created: submissionNEWp.csv")
