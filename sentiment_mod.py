import nltk
import random
import pickle
from statistics import mode
from nltk.classify import ClassifierI
from nltk.tokenize import word_tokenize

# Voting Classifier
class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = [c.classify(features) for c in self._classifiers]
        return mode(votes)

    def confidence(self, features):
        votes = [c.classify(features) for c in self._classifiers]
        choice_votes = votes.count(mode(votes))
        return choice_votes / len(votes)

# Load preprocessed documents
with open("pickled_algo/documents.pickle", "rb") as file:
    documents = pickle.load(file)

# Load word features
with open("pickled_algo/features.pickle", "rb") as file:
    word_features = pickle.load(file)

# Function to find features in a document
def find_features(document):
    words = set(word_tokenize(document))
    return {word: (word in words) for word in word_features}

# Load feature sets
with open("pickled_algo/featureset.pickle", "rb") as file:
    featuresets = pickle.load(file)

# Load trained classifiers
def load_classifier(file_name):
    with open(f"pickled_algo/{file_name}.pickle", "rb") as file:
        return pickle.load(file)

classifiers = {
    "NaiveBayes": load_classifier("naive_bayes"),
    "MultinomialNB": load_classifier("multinomial_nb"),
    "BernoulliNB": load_classifier("bernoulli_nb"),
    "LogisticRegression": load_classifier("logistic_regression"),
    "LinearSVC": load_classifier("linear_svc"),
    # "SGDClassifier": load_classifier("SGDClassifier_classifier"),
}

# Create a voting classifier
voted_classifier = VoteClassifier(
    classifiers["NaiveBayes"],
    classifiers["LinearSVC"],
    classifiers["MultinomialNB"],
    classifiers["BernoulliNB"],
    classifiers["LogisticRegression"],
)

# Sentiment analysis function
def sentiment(text):
    features = find_features(text)
    label = voted_classifier.classify(features)
    confidence = voted_classifier.confidence(features)
    return label, confidence
