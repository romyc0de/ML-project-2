from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Attempt to use MPS on Apple Silicon if available, else CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Load pre-trained BERT tokenizer and model
model_name = "bert-base-uncased"  # You can also use "bert-large-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name).to(device)

# Function to read tweets from a file
def read_tweets(file_path):
    with open(file_path, 'r') as file:
        tweets = [line.strip() for line in file.readlines()]
    return tweets

# Function to extract embeddings for a list of tweets
def extract_embeddings(tweets, batch_size=16):
    print(f"Tokenizing {len(tweets)} tweets...")
    tokens = tokenizer(tweets, padding=True, truncation=True, return_tensors="pt")
    tokens = {key: val.to(device) for key, val in tokens.items()}
    print("Tokenization complete.")

    # Create a DataLoader for batch processing
    dataset = TensorDataset(tokens['input_ids'], tokens['attention_mask'])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Extract embeddings
    print("Starting embeddings extraction...")
    all_embeddings = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Forward pass through BERT
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Extract the [CLS] token embeddings
            batch_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(batch_embeddings.cpu().numpy())

    print("Concatenating embeddings...")
    cls_embeddings = np.concatenate(all_embeddings, axis=0)
    return cls_embeddings

# Process training data
pos_tweets = read_tweets("twitter-datasets/train_pos.txt")
neg_tweets = read_tweets("twitter-datasets/train_neg.txt")
pos_tweets_full = read_tweets("twitter-datasets/train_pos_full.txt")
neg_tweets_full = read_tweets("twitter-datasets/train_neg_full.txt")
train_tweets = pos_tweets + neg_tweets
train_tweets_full = pos_tweets_full + neg_tweets_full
print(f"Total training tweets to process: {len(train_tweets)}")
train_embeddings_full = extract_embeddings(train_tweets_full)
print(f"Total training tweets (full) to process: {len(train_tweets_full)}")

# Save training embeddings
np.save("bert_train_embeddings.npy", train_embeddings)
print("Training embeddings saved as 'bert_train_embeddings.npy'.")
np.save("bert_train_embeddings_full.npy", train_embeddings_full)
print("Training embeddings saved as 'bert_train_embeddings_full.npy'.")

# Process test data
test_tweets = read_tweets("twitter-datasets/test_data.txt")
print(f"Total test tweets to process: {len(test_tweets)}")
test_embeddings = extract_embeddings(test_tweets)

# Save test embeddings
np.save("bert_test_embeddings.npy", test_embeddings)
print("Test embeddings saved as 'bert_test_embeddings.npy'.")
print("Process complete!")
