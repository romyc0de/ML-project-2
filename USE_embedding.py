import tensorflow_hub as hub 
import numpy as np
  

def main():
    # Load pre-trained universal sentence encoder model 
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4") 

    # Function to read tweets from a file
    def read_tweets(file_path):
        with open(file_path, "r") as file:
            return [line.strip() for line in file]

    # Read positive and negative tweets
    pos_tweets = read_tweets("twitter-datasets/train_pos.txt")
    neg_tweets = read_tweets("twitter-datasets/train_neg.txt")
    tweets = pos_tweets + neg_tweets

    embeddings = embed(tweets)

    np.save("USE_embeddings.npy", embeddings) 
    print("Embeddings saved to USE_embeddings.npy")


if __name__ == "__main__":
    main()