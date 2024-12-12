import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Paths to your tweet files
pos_file = "twitter-datasets/train_pos.txt"
neg_file = "twitter-datasets/train_neg.txt"
test_embeddings_file = "embeddings/bert_test_embeddings.npy"  # Saved test embeddings

# Function to read tweets from a file
def read_tweets(file_path):
    with open(file_path, 'r') as file:
        tweets = [line.strip() for line in file.readlines()]
    return tweets

# Read training data
print("Reading training data...")
pos_tweets = read_tweets(pos_file)
neg_tweets = read_tweets(neg_file)

# Combine training tweets and create labels
tweets = pos_tweets + neg_tweets
labels = np.concatenate([
    np.ones(len(pos_tweets), dtype=int),
    -1 * np.ones(len(neg_tweets), dtype=int)
])

# Load the previously generated embeddings for training data
print("Loading training embeddings...")
cls_embeddings_np = np.load("embeddings/bert_embeddings.npy")

# Check consistency
assert len(tweets) == cls_embeddings_np.shape[0], "Mismatch in number of tweets and embeddings."
assert len(labels) == cls_embeddings_np.shape[0], "Mismatch in number of labels and embeddings."

# Split training data to evaluate performance locally
print("Splitting training data for evaluation...")
X_train, X_test, y_train, y_test, tweets_train, tweets_test = train_test_split(
    cls_embeddings_np,
    labels,
    tweets,
    test_size=0.2,
    random_state=42
)

# Train a Logistic Regression classifier
print("Training the classifier...")
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate performance on local test set
print("Evaluating on local test set...")
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.4f}")

#####################################################################
# Load saved test embeddings and predict with progress updates
#####################################################################

# Load the test embeddings
print("Loading saved test embeddings...")
test_embeddings = np.load(test_embeddings_file)

# Ensure we have exactly 10,000 test embeddings
assert test_embeddings.shape[0] == 10000, "Test embeddings file must contain exactly 10,000 embeddings."

# Predict on the test embeddings
print("Predicting on test embeddings...")
y_pred_test = []
for i in tqdm(range(len(test_embeddings)), desc="Predicting test tweets"):
    y_pred_test.append(clf.predict(test_embeddings[i].reshape(1, -1))[0])

# Convert predictions to a numpy array
y_pred_test = np.array(y_pred_test)

# Format predictions as requested:
# Id,Prediction
# 1,-1
# 2,-1
# 3,1
# ...
print("Saving predictions to CSV...")
df_results = pd.DataFrame({
    "Id": np.arange(1, len(y_pred_test) + 1),
    "Prediction": y_pred_test
})

df_results.to_csv("predictions.csv", index=False)
print("Predictions saved to 'predictions.csv' in the requested format.")
