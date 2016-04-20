from __future__ import print_function
import nltk
#nltk.download()
from nltk import TweetTokenizer
import csv
import random
from collections import defaultdict

tokenizer = TweetTokenizer()
csvfile = open('trainingandtestdata/testdata.manual.2009.06.14.csv', 'rb')
reader = csv.reader(csvfile, delimiter=',')
rownum = 0
sentiments = []
tokens = [[]]
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

#Divide into training and test data - randomly allocate 4/5 to training and 1/5 to test
position = []
posPosts = []
negPosts = []
neuPosts = []
numPosPosts = numNegPosts = numNeuPosts = 0
for posts in range(len(sentiments)):
    position.insert(posts,random.randint(0,4))
    if position[posts] != 0:
        if sentiments[posts] == 4:
            posPosts.extend(tokens[posts])
            numPosPosts += 1
        elif sentiments[posts] == 0:
            negPosts.extend(tokens[posts])
            numNegPosts += 1
        else:
            neuPosts.extend(tokens[posts])
            numNeuPosts += 1

wordBag = defaultdict(list)
for token in posPosts:
    wordBag[token].append(1)
for token in negPosts:
    wordBag[token].append(-1)
for token in neuPosts:
    wordBag[token].append(0)

## Calculate Prior
numTestPosts = numPosPosts + numNegPosts + numNeuPosts
priorPos = numPosPosts / float(numTestPosts)
priorNeg = numNegPosts / float(numTestPosts)
priorNeu = numNeuPosts / float(numTestPosts)

# Calculate p(x) and likelihoods for words
wordSen = defaultdict(list)
probWordOcc = {} 
for word in wordBag:
##    print("Word: " + word + ", instances: " + str(len(wordBag[word])) + ", values: ")
##    for value in wordBag[word]:
##            print(value, end=" ")
##    print("Total = " + str(sum(wordBag[word])))
    if len(wordBag[word]) > 2:
        probWordOcc[word] = len(wordBag[word]) / float(numTestPosts)
        posLike = wordBag[word].count(1) / float(len(wordBag[word]))
        negLike = wordBag[word].count(-1) / float(len(wordBag[word]))
        neuLike = wordBag[word].count(0) / float(len(wordBag[word]))
        wordSen[word] = [posLike,negLike,neuLike]
##    print("word: " + word + ", word sentiment: " + str(wordSen[word]))
    
#testing phase
corrects = []
for posts in range(len(sentiments)):
    if position[posts] == 0:
        postProbs = [1,1,1]
        for token in tokens[posts]:
            if token in wordSen:
                for i in range(3):
                    postProbs[i] *= (wordSen[token][i] * priorPos / float(probWordOcc[token]))
##                print(postProbs)
        predictedIndex = postProbs.index(max(postProbs))
        if predictedIndex == 0:
            corrects.append(sentiments[posts] == 4)
##            print("postive: " + str(sentiments[posts] == 4))
        elif predictedIndex == 1:
            corrects.append(sentiments[posts] == 0)
##            print("negative: " + str(sentiments[posts] == 0))
        else:
            corrects.append(sentiments[posts] == 2)
##            print("neutral: " + str(sentiments[posts] == 2))
            
##print(corrects)
print("The number of correctly predicted posts is " + str(sum(corrects)) + " out of " + str(len(corrects)))
print("That's about " + str(sum(corrects) * 100 / float(len(corrects))) + "%")
