import gensim
from gensim.models import Word2Vec
import numpy as np

def main():
    # Function to read tweets from a file
    def read_tweets(file_path):
        with open(file_path, 'r') as file:
            tweets = [line.strip() for line in file.readlines()]
        return tweets

    # Read positive and negative tweets from the respective files
    pos_tweets = read_tweets("twitter-datasets/train_pos.txt")
    neg_tweets = read_tweets("twitter-datasets/train_neg.txt")

    # Combine both lists
    tweets = pos_tweets + neg_tweets

    # Tokenize tweets (split by space)
    tokenized_tweets = [tweet.split() for tweet in tweets]

    # Train Word2Vec model
    # Here, we use a Skip-Gram model ('sg=1') and a window size of 5 (can be tuned)
    model = Word2Vec(sentences=tokenized_tweets, vector_size=100, window=5, min_count=1, sg=1)

    # Now, to get embeddings for a specific word, you can use the model:
    word_embeddings = model.wv

    # Optionally save embeddings to a numpy file
    word_vectors = word_embeddings.vectors  # This gives you the word vectors (numpy array)
    np.save("word2vec_embeddings.npy", word_vectors)

    print("Word2Vec embeddings saved as word2vec_embeddings.npy")


if __name__ == "__main__":
    main()
