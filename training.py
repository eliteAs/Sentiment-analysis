import pandas as pd
import nltk
from nltk import word_tokenize
import random
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import pickle
from statistics import mode
from nltk.classify import ClassifierI
from collections import Counter
import re
from nltk.corpus import stopwords

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self.classifiers = classifiers

    def classify(self, features):
        votes = [c.classify(features) for c in self.classifiers]
        return mode(votes)
    
    def confidence(self, features):
        votes = [c.classify(features) for c in self.classifiers]
        choice_votes = votes.count(mode(votes))
        return choice_votes / len(votes)

data = pd.read_csv("dataset.csv")

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetical characters
    return text.lower()

data['review'] = data['review'].apply(clean_text)

# Extract reviews and sentiments
all_reviews = [(review, sentiment) for review, sentiment in zip(data['review'], data['sentiment'])]

# Shuffle data for randomness
random.shuffle(all_reviews)

# Efficient word extraction and stopword removal
def extract_adjectives(reviews):
    all_words = []
    documents = []
    allowed_word_types = {"J"}  # Adjectives
    stop_words = set(stopwords.words('english'))
    
    for review, label in reviews:
        words = word_tokenize(review)
        pos = nltk.pos_tag(words)
        documents.append((review, label))
        
        # Removing stopwords and extract adjectives
        filtered_words = [w[0].lower() for w in pos if w[1][0] in allowed_word_types and w[0].lower() not in stop_words]
        all_words.extend(filtered_words)
        
    return documents, all_words

documents, all_words = extract_adjectives(all_reviews)

# Save documents
with open("pickled_algo/documents.pickle", "wb") as save_doc:
    pickle.dump(documents, save_doc)


# Extract the most frequent words
all_words = Counter(all_words)
word_features = [word for word, freq in all_words.most_common(5000)]


# Save word features
with open("pickled_algo/features.pickle", "wb") as save_word_features:
    pickle.dump(word_features, save_word_features)


# Feature extraction function
def find_features(document):
    words = set(word_tokenize(document))
    return {word: (word in words) for word in word_features}


# Create feature sets
featuresets = [(find_features(review), sentiment) for (review, sentiment) in documents]


# Save feature sets
with open("pickled_algo/featureset.pickle", "wb") as save_featureset:
    pickle.dump(featuresets, save_featureset)


# Split into training and testing sets
train_size = int(0.8 * len(featuresets))
training_set = featuresets[:train_size]
testing_set = featuresets[train_size:]


# Train classifiers
def train_and_save_classifier(classifier, training_set, file_name):
    classifier.train(training_set)
    accuracy = nltk.classify.accuracy(classifier, testing_set) * 100
    print(f"{file_name} accuracy: {accuracy:.2f}%")
    with open(f"pickled_algo/{file_name}.pickle", "wb") as save_file:
        pickle.dump(classifier, save_file)
    return classifier

classifiers = [
    (lambda: nltk.NaiveBayesClassifier.train(training_set), "naive_bayes"),
    (lambda: SklearnClassifier(MultinomialNB()).train(training_set), "multinomial_nb"),
    (lambda: SklearnClassifier(BernoulliNB()).train(training_set), "bernoulli_nb"),
    (lambda: SklearnClassifier(LogisticRegression()).train(training_set), "logistic_regression"),
    (lambda: SklearnClassifier(LinearSVC()).train(training_set), "linear_svc"),
]

trained_classifiers = []
for cls_func, cls_name in classifiers:
    classifier = cls_func()  # Call the lambda to train and retrieve the classifier
    trained_classifiers.append(train_and_save_classifier(classifier, training_set, cls_name))


# Voting classifier
voted_classifier = VoteClassifier(*trained_classifiers)
print("Voted classifier accuracy:", nltk.classify.accuracy(voted_classifier, testing_set) * 100)

# Sentiment analysis function
def sentiment_analysis(text):
    features = find_features(text)
    return voted_classifier.classify(features), voted_classifier.confidence(features)

