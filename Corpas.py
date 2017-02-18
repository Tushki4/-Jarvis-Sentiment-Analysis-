from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize

syns = wordnet.synsets("robust")

print(syns[0])


#### gets just the word
print(syns[0].lemmas()[0].name())

## definition of the word

print(syns[0].definition())

## examples

print(syns[0].examples())

synonyms = []
antonyms = []


for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))




######## you can use this to rewrite this... 

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("boat.n.01")

print(w1.wup_similarity(w2))


w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("car.n.01")

print(w1.wup_similarity(w2))


w1 = wordnet.synset("cat.n.01")
w2 = wordnet.synset("dog.n.01")

print(w1.wup_similarity(w2))










































