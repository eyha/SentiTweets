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
from math import pow
from functools import reduce
import operator

class negClauseLearner:
    offsets = [[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]
    # Checks for negations in sentences and creates a list of affected words
    def negCheck(self,tokens):
        positives = []
        negatives = []
        currentSentence = []
        negative = False
        negativeSenNum = 0
        for token in tokens:
            currentSentence.append(token)
            if(token in negations):
                negative = True
            elif(re.match('[.!?]+',token)):
                if negative:
                    negatives.extend(currentSentence)
                    # print(currentSentence)
                    currentSentence = []
                    negative = False
                    negativeSenNum += 1
                else:
                    positives.extend(currentSentence)
                    currentSentence = []
        if negative:
            negatives.extend(currentSentence)
            currentSentence = []
            negative = False
        else:
            positives.extend(currentSentence)
            currentSentence = []
        # print(positives)
        # print(negatives)
        return (positives,negatives,negativeSenNum)

    #tokenizer = TweetTokenizer()
    #csvfile = open('trainingandtestdata/testdata.manual.2009.06.14.csv', 'rb')
    #reader = csv.reader(csvfile, delimiter=',')
    #rownum = 0
    #sentiments = []
    #tokens = [[]]
    #for row in reader:
    #    colnum = 0
    #    for col in row:
    #        if colnum == 0:
    #            sentiments.insert(rownum,int(col))
    #        if colnum == 5:
    #            raw = col #.read().decode('utf8')
    #            tokens.insert(rownum,tokenizer.tokenize(raw))
    ##            print("tokens contents:", end='')
    ##            for word in tokens[rownum]:
    ##                print(word, end = " ")
    ##            print()
     #       colnum += 1
     #   rownum += 1
    #csvfile.close()

    def negTesting(self, sentiments, tokens):
        #Divide into training and test data - randomly allocate 4/5 to training and 1/5 to test
        position = []
        posPosts = []
        negPosts = []
        neuPosts = []
        for posts in range(0,len(sentiments)):
            position.insert(posts,random.randint(0,4))
            if position[posts] != 0:
                posNegs = self.negCheck(tokens[posts])
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
        for postPosition in range(0,len(sentiments)):
            if position[postPosition] == 0:
                score = 0
                for token in tokens[postPosition]:
                    if token in wordSen:
                        score += float(wordSen[token])
        ##                print(score)
                if score >= 2.5:
                    corrects.append(sentiments[postPosition] == 4)
        ##            print("postive: " + str(sentiments[postPosition] == 4))
                elif score <= -2.5:
                    corrects.append(sentiments[postPosition] == 0)
        ##            print("negative: " + str(sentiments[postPosition] == 0))
                else:
                    corrects.append(sentiments[postPosition] == 2)
        ##            print("neutral: " + str(sentiments[postPosition] == 2))
                
        #print(corrects)
        numCorrects = sum(corrects)
        numTestCases = len(corrects)
        accuracy = float(numCorrects * 100/float(numTestCases))
#       print("The number of correctly predicted posts is " + str(numCorrects) + " out of " + str(numTestCases))
        return accuracy
#print ("Total Accuracy: " + str(negTesting(sentiments)) + "%")

    def trainNeg(self,mainTest,sentiments,tokens):
        predicted = [[]]
        postNumber = 0
        differences = [
            [[0],[],[]],
            [[],[0],[]],
            [[],[],[0]]]
        for postPosition in range(len(sentiments)):
            if mainTest.position[postPosition] != 0:
                postProbs = [1,1,1]
                for i in range(3):
                    for token in tokens[postPosition]:
                        if token in mainTest.wordSen:
                        # print ("for token " + token)
                            # print("The " + str(i) + " likelihood is " + str(mainTest.wordSen[token][i]))
                            # print(", the " + str(i) + " prior is " + str(mainTest.priors[i]))
                            # print("and the " + str(i) + " evidence is " + str(mainTest.evidences[token][i]))
                            if mainTest.wordSen[token][i] != 0:
                                postProbs[i] *= mainTest.wordSen[token][i]
                            ## If mainTest.evidences[token][i] is 0, then you would want to divide by 1, so do nothing
                            if mainTest.evidences[token][i] != 0:
                                postProbs[i] /= float(mainTest.evidences[token][i])
                postProbs[i] *= mainTest.priors[i]
                # print("Calculated probability is " + str(postProbs))
                predictedIndex = postProbs.index(max(postProbs))
                predicted.insert(postPosition,postProbs)
                ## Check if prediction is correct
                ## Calculate the sentiment's corresponding index
                actualIndex = pow(sentiments[postPosition],2) + 1
                actualIndex = int(actualIndex % 5 - 2)
                actualIndex = int(abs(actualIndex))
                # fabs(((pow(sentiments[postPosition]),2)+1) % 5 - 2)
                # for i in [4,0,2]:
                    # test = pow(i,2) + 1
                    # test = int(test % 5 - 2)
                    # print (abs(test))
                    # print (fabs((((pow(i,2) + 1) % 5) - 2)))
                if actualIndex != predictedIndex:
                    divided = self.negCheck(tokens[postPosition])
                    if divided[2] != 0:
                        print("Predicted place was " + str(predictedIndex) + " with " + str(postProbs[predictedIndex])) 
                        print("Actual was " + str(actualIndex) + " with " + str(postProbs[actualIndex]))
                        ## Offset for sentence is the difference between the predicted sentiment and the actual, divided by the number of negative sentences
                        differences[actualIndex][predictedIndex].append((postProbs[predictedIndex] - postProbs[actualIndex])/float(divided[2]))
        # print("First column: " + str(differences[0]))
        # print("Second column: " + str(differences[1]))
        # print("Third column: " + str(differences[2]))
        for column in range(3):
            for row in range(3):
                # print("Column: " + str(column) + ", row: " + str(row))
                if len(differences[column][row]) != 0:
                    offset = sum(differences[column][row]) / float(len(differences[column][row]))
                    # print(offset)
                    self.offsets[column][row] = offset
                else:
                    self.offsets[column][row] = 0.0
        print(self.offsets)
            # print(differences)
                # elif sentiments[postPosition] == 0:
                    # if predictedIndex != 1:
                        # differences[1][predictedIndex].append(postProbs[predictedIndex] - postProbs[1])
                        # divided = negCheck(tokens(postPosition))
                # else:
                    # if predictedIndex != 2:
                        # differences[2][predictedIndex].append(postProbs[predictedIndex] - postProbs[2])
                        # divided = self.negCheck(tokens[postPosition])

    # def suf(self):
        