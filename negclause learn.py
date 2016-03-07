from __future__ import print_function
import nltk
#nltk.download()
from nltk import TweetTokenizer
import csv
import random
from collections import defaultdict
import re
import negation
from negation import negations 

# Checks for negations in sentences and creates a list of affected words
def negCheck(tokens):
    positives = []
    negatives = []
    currentSentence = []
    negative = False
    for token in tokens:
        if(token in negations):
            negative = True
        elif(re.match('[.!?]+',token)):
            if negative:
                negatives.extend(currentSentence)
                currentSentence = []
                negative = False
            else:
                positives.extend(currentSentence)
                currentSentence = []
    #if not currentSentence:
    if negative:
        negatives.extend(currentSentence)
        currentSentence = []
        negative = False
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

def negTesting(sentiments):
    #Divide into training and test data - randomly allocate 4/5 to training and 1/5 to test
    position = []
    posPosts = []
    negPosts = []
    neuPosts = []
    for posts in range(0,len(sentiments)):
        position.insert(posts,random.randint(0,4))
        if position[posts] != 0:
            posNegs = negCheck(tokens[posts])
            if sentiments[posts] == 4:
                posPosts.extend(posNegs[0])
                negPosts.extend(posNegs[1])
            elif sentiments[posts] == 0:
                negPosts.extend(posNegs[1])
                posPosts.extend(posNegs[0])
            else:
                neuPosts.extend(posNegs[0])
                neuPosts.extend(posNegs[1])

    wordBag = defaultdict(list)
    for token in posPosts:
        wordBag[token].append(5)
    for token in negPosts:
        wordBag[token].append(-5)

    for token in neuPosts:
        wordBag[token].append(0)

    wordSen = {}
    for word in wordBag:
    ##    print("Word: " + word + ", instances: " + str(len(wordBag[word])) + ", values: ")
    ##    for value in wordBag[word]:
    ##            print(value, end=" ")
    ##    print("Total = " + str(sum(wordBag[word])))
        wordSen[word] = sum(wordBag[word]) / float(len(wordBag[word]))
    ##    print("word: " + word + ", word sentiment: " + str(wordSen[word]))
        
    #testing phase
    corrects = []
    for posts in range(0,len(sentiments)):
        if position[posts] == 0:
            score = 0
            for token in tokens[posts]:
                if token in wordSen:
                    score += float(wordSen[token])
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
    numCorrects = sum(corrects)
    numTestCases = len(corrects)
    accuracy = float(numCorrects * 100/float(numTestCases))
    print("The number of correctly predicted posts is " + str(numCorrects) + " out of " + str(numTestCases))
    return accuracy
    
print ("Total Accuracy: " + str(negTesting(sentiments)) + "%")
