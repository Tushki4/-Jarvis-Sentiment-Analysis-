import nltk
import random
#from nltk.corpus import movie_reviews
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize

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
        

short_pos = open("positive.txt", "r").read()
short_neg = open("negative.txt", "r").read()


all_words = []
documents = []

allowed_word_types = ["J"]

for p in short_pos.split('\n'):
    documents.append( (p, "positive") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for p in short_neg.split('\n'):
    documents.append( (p, "negative") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

save_documents = open("documents.pickle", "wb")
pickle.dump(documents, save_documents)
save_documents.close()


all_words = nltk.FreqDist(all_words)


### limit on the number of words.upto 3000 words, top 15 included dashes, periods, 3000 would have sufficent words to classify into +ve and -ve

word_features = list(all_words.keys())[:5000]

save_word_features = open("word_features5k.pickle" , "wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()


def find_features(document):
    words = word_tokenize(document)  ##every single word will be included in the set
    features = {}
    for w in word_features:
        features[w] = (w in words)  # creates a boolean with either true or false

    return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]
# find features in the categories... converting it into anothe reviews with true or false, whether top 3000 words are present in the reviews


random.shuffle(featuresets)

training_set = featuresets[:5000]

testing_set = featuresets[5000:]

classifier = nltk.NaiveBayesClassifier.train(training_set)

print("Accuracy :" , (nltk.classify.accuracy(classifier, testing_set)))
#classifier.show_most_informative_features(15)

save_classifier = open("naivebayes5k.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

### Multinomial Naive Bayes
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classfier:", (nltk.classify.accuracy(classifier, testing_set)))

save_classifier = open("MNB_classifier5k.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

##### Gaussian Naive Bayes
##Gaussian_NB_classifier = SklearnClassifier(GaussianNB())
##Gaussian_NB_classifier.train(training_set)
##print("GNB_classfier:", (nltk.classify.accuracy(Gaussian_NB_classifier, testing_set)))

### Bernoulli Naive Bayes
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BNB_classfier:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)))

save_classifier = open("BernoulliNB_classifier5k.pickle", "wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

#LogisticRegression, SGDClassifier
#SVC, LinearSVC, NuSVC

#Logistic_Classifier
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)))

save_classifier = open("LogisticRegression_classifier5k.pickle", "wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()

#SGD
SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set)))

save_classifier = open("SGDClassifier_classifier5k.pickle", "wb")
pickle.dump(SGDClassifier_classifier, save_classifier)
save_classifier.close()
###SVC
##SVC_classifier = SklearnClassifier(SVC())
##SVC_classifier.train(training_set)
##print("SVC_classifier:", (nltk.classify.accuracy(SVC_classifier, testing_set)))
##
#LinearSVC_classifier
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set)))

save_classifier = open("LinearSVC_classifier5k.pickle", "wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()

#NuSVC_classifier
NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier:", (nltk.classify.accuracy(NuSVC_classifier, testing_set)))

save_classifier = open("NuSVC_classifier5k.pickle", "wb")
pickle.dump(NuSVC_classifier, save_classifier)
save_classifier.close()



    

