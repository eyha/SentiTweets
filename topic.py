from __future__ import print_function
import nltk
#nltk.download()
from nltk import TweetTokenizer
import csv
import random
from collections import defaultdict
import hashTags

tokenizer = TweetTokenizer()
csvfile = open('trainingandtestdata/testdata.manual.2009.06.14.csv', 'rb')
reader = csv.reader(csvfile, delimiter=',')
rownum = 0
sentiments = []
tokens = [[]]
realTopics = []
for row in reader:
    colnum = 0
    for col in row:
        # The column with the actual sentiment of the post
        if colnum == 0:
            sentiments.insert(rownum,int(col))
        # The topic of the post
        if colnum == 3:
            realTopics.insert(rownum,col)
        # The content of the post
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
for posts in range(0,len(sentiments)):
    position.insert(posts,random.randint(0,4))
    if position[posts] != 0:
        if sentiments[posts] == 4:
            posPosts.extend(tokens[posts])
        elif sentiments[posts] == 0:
            negPosts.extend(tokens[posts])
        else:
            neuPosts.extend(tokens[posts])

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
print("The number of correctly predicted posts is " + str(sum(corrects)) + " out of " + str(len(corrects)))