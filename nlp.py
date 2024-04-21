import nltk
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy

nltk.download('movie_reviews')
nltk.download('punkt')

def extract_features(words):
    return dict([(word, True) for word in words])

positive_reviews = [(extract_features(movie_reviews.words(fileids=[f])), 'Positive') for f in movie_reviews.fileids('pos')]
negative_reviews = [(extract_features(movie_reviews.words(fileids=[f])), 'Negative') for f in movie_reviews.fileids('neg')]

split_index = int(0.8 * len(positive_reviews))
train_set = positive_reviews[:split_index] + negative_reviews[:split_index]
test_set = positive_reviews[split_index:] + negative_reviews[split_index:]

classifier = NaiveBayesClassifier.train(train_set)

accuracy = nltk_accuracy(classifier, test_set)
print("Accuracy:", accuracy)

new_review = "This movie was fantastic! I loved every minute of it."
words = word_tokenize(new_review)
features = extract_features(words)
print("Sentiment of the review:", classifier.classify(features))