# Author: Aaron Dhillon and Ben Varkey
# Description: Generate word2vec embeddings for sentiment analysis

import numpy as np
import gensim.downloader as api
from SentimentNaiveBayes import data_reader

# Load word2vec model
print("Loading word2vec-google-news-300 model...")
wv = api.load('word2vec-google-news-300')
print("Model loaded successfully!")

# Function to compute average word2vec for a sentence
def sentence_to_mean_vector(sentence, wv):
    words = sentence.lower().split()
    vectors = [wv[word] for word in words if word in wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(300)  # Handle case where none of the words are in the model

# Function to prepare dataset: returns embeddings and labels
def prepare_data(path, wv):
    sentences, labels = data_reader(path)
    X = np.array([sentence_to_mean_vector(sent, wv) for sent in sentences])
    y = np.array(labels)
    return X, y

if __name__ == "__main__":
    # Paths
    train_path = "data-sentiment/train/"
    test_path = "data-sentiment/test/"

    # Create training and test data embeddings
    X_train, y_train = prepare_data(train_path, wv)
    X_test, y_test = prepare_data(test_path, wv)

    # Save the results as .npy files to avoid recomputation
    np.save("X_train.npy", X_train)
    np.save("y_train.npy", y_train)
    np.save("X_test.npy", X_test)
    np.save("y_test.npy", y_test)

    print("Sentence embeddings saved as .npy files.")
