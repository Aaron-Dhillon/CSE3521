#Author:Ben Varkey and Aaron Dhillon
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math

def file_reader(file_path, label):
    list_of_lines = []
    list_of_labels = []

    for line in open(file_path):
        line = line.strip()
        if line=="":
            continue
        list_of_lines.append(line)
        list_of_labels.append(label)

    return (list_of_lines, list_of_labels)


def data_reader(source_directory):
    positive_file = source_directory+"Positive.txt"
    (positive_list_of_lines, positive_list_of_labels)=file_reader(file_path=positive_file, label=1)

    negative_file = source_directory+"Negative.txt"
    (negative_list_of_lines, negative_list_of_labels)=file_reader(file_path=negative_file, label=-1)

    neutral_file = source_directory+"Neutral.txt"
    (neutral_list_of_lines, neutral_list_of_labels)=file_reader(file_path=neutral_file, label=0)

    list_of_all_lines = positive_list_of_lines + negative_list_of_lines + neutral_list_of_lines
    list_of_all_labels = np.array(positive_list_of_labels + negative_list_of_labels + neutral_list_of_labels)

    return list_of_all_lines, list_of_all_labels


def evaluate_predictions(test_set,test_labels,trained_classifier):
    correct_predictions = 0
    predictions_list = []
    prediction = -1
    for dataset,label in zip(test_set, test_labels):
        probabilities = trained_classifier.predict(dataset)
        if probabilities[0] >= probabilities[1] and probabilities[0] >= probabilities[-1]:
            prediction = 0
        elif  probabilities[1] >= probabilities[0] and probabilities[1] >= probabilities[-1]:
            prediction = 1
        else:
            prediction=-1
        if prediction == label:
            correct_predictions += 1
            predictions_list.append("+")
        else:
            predictions_list.append("-")
    
    print("Total Sentences: ", len(test_labels))
    print("Predicted correctly: ", correct_predictions)
    print("Accuracy: {}%".format(round(correct_predictions/len(test_labels)*100,5)))

    return predictions_list, round(correct_predictions/len(test_labels)*100)


class NaiveBayesClassifier(object):
    def __init__(self, n_gram=1, printing=False):
        self.prior = []
        self.conditional = []
        self.V = []
        self.n = n_gram
        self.BOW=[]
        self.classCounts=[]
        self.D=0
        self.N=0
        self.labelmap={}

    def word_tokenization_dataset(self, training_sentences):
        training_set = list()
        for sentence in training_sentences:
            cur_sentence = list()
            for word in sentence.split(" "):
                cur_sentence.append(word.lower())
            training_set.append(cur_sentence)
        return training_set

    def word_tokenization_sentence(self, test_sentence):
        cur_sentence = list()
        for word in test_sentence.split(" "):
            cur_sentence.append(word.lower())
        return cur_sentence

    def compute_vocabulary(self, training_set):
        vocabulary = set()
        for sentence in training_set:
            for word in sentence:
                vocabulary.add(word)
        V_dictionary = dict()
        dict_count = 0
        for word in vocabulary:
            V_dictionary[word] = int(dict_count)
            dict_count += 1
        return V_dictionary

    def to_BOW_sentence(self, sentence):
        n = len(self.V)
        bow = [0]*n
        for word in self.V.keys():
            if word in sentence:
                bow[self.V[word]] = 1
        return bow

    def to_BOW_array(self, sentences):
        n = len(sentences)
        bow_array = [0]*n
        for i, sentence in enumerate(sentences):
            bow_array[i] = self.to_BOW_sentence(sentence)
        return bow_array

    def train(self, training_sentences, training_labels):
        N_sentences = len(training_sentences)
        training_set = self.word_tokenization_dataset(training_sentences)
        self.V = self.compute_vocabulary(training_set)
        self.BOW = self.to_BOW_array(training_set)

        counts = {0:0.0, 1:0.0, -1:0.0}
        for i in range(N_sentences):
            counts[training_labels[i]] += 1.0

        self.prior = {0:counts[0]/N_sentences, 1: counts[1]/N_sentences, -1: counts[-1]/N_sentences}

        self.conditional = {0: [0.0]*len(self.V), 1: [0.0]*len(self.V), -1: [0.0]*len(self.V)}
        for i in range(N_sentences):
            for j in range(len(self.V)):
                if self.BOW[i][j] == 1:
                    self.conditional[training_labels[i]][j] += (1/counts[training_labels[i]])

    def predict(self, test_sentence):
        label_probability = {0: 0, 1: 0, -1: 0}
        test_sentence = self.word_tokenization_sentence(test_sentence)
        epsilon = 1e-10
        bow = self.to_BOW_sentence(test_sentence)

        for label in label_probability.keys():
            prob = np.log(self.prior[label])
            for i in range(len(self.V)):
                p = self.conditional[label][i]
                if bow[i] == 1:
                    if p < epsilon:
                        p = epsilon
                    prob += np.log(p)
                else:
                    if p == 1.0:
                        p = p - epsilon
                    prob += np.log(1-p)
            label_probability[label] = prob

        return label_probability

def evaluate_naive_bayes_subset(training_sentences, training_labels, test_sentences, test_labels, subset_sizes):
    results = {}
    data = list(zip(training_sentences, training_labels))
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    data = [data[i] for i in shuffled_indices]
    shuffled_sentences, shuffled_labels = zip(*data)

    for subset_size in subset_sizes:
        print(f"\nTraining with subset size {subset_size}")
        subset_sentences = shuffled_sentences[:subset_size]
        subset_labels = shuffled_labels[:subset_size]

        nb = NaiveBayesClassifier(n_gram=1)
        nb.train(subset_sentences, subset_labels)

        _, acc = evaluate_predictions(test_sentences, test_labels, nb)
        results[subset_size] = acc

    return results

TASK = 'test_subset'  # 'train'  'test' 'test_subset'

if TASK == 'train':
    train_folder = "data-sentiment/train/"
    training_sentences, training_labels = data_reader(train_folder)
    NBclassifier = NaiveBayesClassifier(n_gram=1)
    NBclassifier.train(training_sentences, training_labels)
    f = open('classifier.pkl', 'wb')
    pickle.dump(NBclassifier, f)
    f.close()

if TASK == 'test':
    test_folder = "data-sentiment/test/"
    test_sentences, test_labels = data_reader(test_folder)
    f = open('classifier.pkl', 'rb')
    NBclassifier = pickle.load(f)
    f.close()
    results, acc = evaluate_predictions(test_sentences, test_labels, NBclassifier)

if TASK == 'test_subset':
    train_folder = "data-sentiment/train/"
    test_folder = "data-sentiment/test/"
    training_sentences, training_labels = data_reader(train_folder)
    test_sentences, test_labels = data_reader(test_folder)
    subset_sizes = [25, 50, 150, 200, 300]
    acc_results = evaluate_naive_bayes_subset(training_sentences, training_labels, test_sentences, test_labels, subset_sizes)

    print("\nAccuracy results by subset size:")
    for size, acc in acc_results.items():
        print(f"Subset Size {size}: {acc}%")
