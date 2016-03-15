from __future__ import print_function
import nltk
#nltk.download()
from nltk import TweetTokenizer
import csv
import random
from collections import defaultdict
import re
import negation

# Checks for negations in sentences and creates a list of affected words
def negCheck(tokens):
    positives = []
    negatives = []
    currentSentence = []
    negative = false
    negWords = []
    for word in range(0,len(negation)):
        negWords[word] = re.compile(negation[word])
    for token in tokens:
        for neg in negWords:
            if neg.match(token)
        if(token in negation):
            negative = true
        elif(re.match('[.!?]+')):
            if negative:
                negatives.extend(currentSentence)
                currentSentence = []
                negative = false
            else:
                positives.extend(currentSentence)
                currentSentence = []
    if not currentSentence:
    elif negative:
        negatives.extend(currentSentence)
        currentSentence = []
        negative = false
    else:
        positives.extend(currentSentence)
        currentSentence = []
    return (positives,negatives)

tokenizer = TweetTokenizer()
csvfile = open('trainingandtestdata/testdata.manual.2009.06.14.csv', 'rb')
reader = csv.reader(csvfile, delimiter=',')
rownum = 0
sentiments = []
tokens = [[]]
bigrams = [[[]]]
for row in reader:
    colnum = 0
    for col in row:
        if colnum == 0:
            sentiments.insert(rownum,int(col))
        if colnum == 5:
            raw = col #.read().decode('utf8')
            tokens.insert(rownum,tokenizer.tokenize(raw))
##            print("tokens contents:", end='')
##            for word in tokens[rownum]:
##                print(word, end = " ")
##            print()
        colnum += 1
    rownum += 1
csvfile.close()

## convert to bigrams
for post in range(0,len(tokens)):
    bigrammedPost = [[]]
    for first in range(0,len(tokens[post])-1):        
        bigrammedPost.insert(first,[tokens[post][first],tokens[post][first+1]])
    bigrams.insert(post,bigrammedPost)

#Divide into training and test data - randomly allocate 4/5 to training and 1/5 to test
position = []
posPosts = []
negPosts = []
neuPosts = []
for posts in range(0,len(sentiments)):
    position.insert(posts,random.randint(0,4))
    if position[posts] != 0:
        posNegs = negCheck(bigrams[posts])
        if sentiments[posts] == 4:
            posPosts.extend(bigrams[posts])
        elif sentiments[posts] == 0:
            negPosts.extend(bigrams[posts])
        else:
            neuPosts.extend(bigrams[posts])

wordBag = defaultdict(list)
for bigram in posPosts:
##    print(bigram)
    if(len(bigram) > 0):
        bigramTuple = bigram[0] + bigram[1]
        wordBag[bigramTuple].append(5)
for bigram in negPosts:
    if(len(bigram) > 0):
        bigramTuple = bigram[0] + bigram[1]
        wordBag[bigramTuple].append(-5)
for bigram in neuPosts:
    if(len(bigram) > 0):
        bigramTuple = bigram[0] + bigram[1]
        wordBag[bigramTuple].append(0)

wordSen = {}
for word in wordBag:
##    print("Word: " + word + ", instances: " + str(len(wordBag[word])) + ", values: ")
##    for value in wordBag[word]:
##            print(value, end=" ")
##    print("Total = " + str(sum(wordBag[word])))
    wordSen[word] = sum(wordBag[word]) / float(len(wordBag[word]))
##    print("word: " + word + ", word sentiment: " + str(wordSen[word]))

##print wordSen

#testing phase
corrects = []
for posts in range(0,len(sentiments)):
    if position[posts] == 0:
        score = 0
        for bigram in bigrams[posts]:
            if bigram in wordSen.keys():
                score += float(wordSen[bigram])
##                print(score)
        if score >= 2.5:
            corrects.append(sentiments[posts] == 4)
##            print("postive: " + str(sentiments[posts] == 4))
        elif score <= -2.5:
            corrects.append(sentiments[posts] == 0)
##            print("negative: " + str(sentiments[posts] == 0))
        else:
            corrects.append(sentiments[posts] == 2)
##            print("neutral: " + str(sentiments[posts] == 2))
            
##print(corrects)
print("The number of correctly predicted posts is " + str(sum(corrects)) + " out of " + str(len(corrects)))
