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

    def negTesting(self, mainTest, sentiments, tokens, postProbs):
        ## testing phase
        for postIndex in range(len(sentiments)):
            if mainTest.position[postIndex] == 0:
                predictedIndex = postProbs[postIndex].index(max(postProbs[postIndex]))
                ## Check for negations in the file to apply offsets
                divided = self.negCheck(tokens[postIndex])
                ## If any clauses with a negation is found
                if divided[2] != 0:
                    for i in range(3):
                        postProbs[postIndex][i] += self.offsets[predictedIndex][i]
        return postProbs

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
                # for i in [4,0,2]:
                    # test = pow(i,2) + 1
                    # test = int(test % 5 - 2)
                    # print (abs(test))
                if actualIndex != predictedIndex:
                    divided = self.negCheck(tokens[postPosition])
                    if divided[2] != 0:
                        # print("Predicted place was " + str(predictedIndex) + " with " + str(postProbs[predictedIndex])) 
                        # print("Actual was " + str(actualIndex) + " with " + str(postProbs[actualIndex]))
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
        # print(self.offsets)        