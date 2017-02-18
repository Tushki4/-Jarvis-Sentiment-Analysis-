import nltk
import random
from nltk.corpus import movie_reviews
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        
    
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)

        return conf
        

##documents = []
##
##for category in movie_reviews.categories():
##    for fileid in movie_reviews.fileids(category):
##        documents.append(list(movie_reviews.words(fileid)), category)


documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]


random.shuffle(documents)

#print(documents[1])

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

#print(all_words.most_common(15))
#print(all_words["stupid"])


### limit on the number of words.upto 3000 words, top 15 included dashes, periods, 3000 would have sufficent words to classify into +ve and -ve

word_features = list(all_words.keys())[:3000]


def find_features(document):
    words = set(document)  ##every single word will be included in the set
    features = {}
    for w in word_features:
        features[w] = (w in words)  # creates a boolean with either true or false

    return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]
# find features in the categories... converting it into anothe reviews with true or false, whether top 3000 words are present in the reviews

training_set = featuresets[:1900]

testing_set = featuresets[1900:]

#classifier = nltk.NaiveBayesClassifier.train(training_set)


classifier_f = open("naive_bayes.picke", "rb")

classifier = pickle.load(classifier_f)

classifier_f.close()

print("Accuracy :" , (nltk.classify.accuracy(classifier, testing_set)))

### Multinomial Naive Bayes
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classfier:", (nltk.classify.accuracy(classifier, testing_set)))

##### Gaussian Naive Bayes
##Gaussian_NB_classifier = SklearnClassifier(GaussianNB())
##Gaussian_NB_classifier.train(training_set)
##print("GNB_classfier:", (nltk.classify.accuracy(Gaussian_NB_classifier, testing_set)))

### Bernoulli Naive Bayes
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BNB_classfier:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)))

#LogisticRegression, SGDClassifier
#SVC, LinearSVC, NuSVC

#Logistic_Classifier
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)))

#SGD
SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set)))

###SVC
##SVC_classifier = SklearnClassifier(SVC())
##SVC_classifier.train(training_set)
##print("SVC_classifier:", (nltk.classify.accuracy(SVC_classifier, testing_set)))
##
#LinearSVC_classifier
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set)))

#NuSVC_classifier
NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier:", (nltk.classify.accuracy(NuSVC_classifier, testing_set)))




voted_classifier = VoteClassifier(classifier, MNB_classifier, BernoulliNB_classifier, LogisticRegression_classifier, SGDClassifier_classifier, LinearSVC_classifier, NuSVC_classifier) 

print("voted_classifier:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

print("Classification:" , voted_classifier.classify(testing_set[0][0]), "Confidence %", voted_classifier.confidence(testing_set[0][0]))
print("Classification:" , voted_classifier.classify(testing_set[1][0]), "Confidence %", voted_classifier.confidence(testing_set[1][0]))
print("Classification:" , voted_classifier.classify(testing_set[2][0]), "Confidence %", voted_classifier.confidence(testing_set[2][0]))
print("Classification:" , voted_classifier.classify(testing_set[3][0]), "Confidence %", voted_classifier.confidence(testing_set[3][0]))
print("Classification:" , voted_classifier.classify(testing_set[4][0]), "Confidence %", voted_classifier.confidence(testing_set[4][0]))


classifier.show_most_informative_features(15)





##save_classifier = open("naive_bayes.picke", "wb")
##
##pickle.dump(classifier, save_classifier)
##
##save_classifier.close()










