from __future__ import print_function
import nltk
#nltk.download()
from nltk import TweetTokenizer
import csv
import random
from collections import defaultdict
import re
from itertools import ifilterfalse

class emoticonLearner:
    offsets = {}
    
    def findEmotes(self,wordBag):
        ## match all tokens that have more than one special characters - need to handle HTML special characters
        ## - can't handle ones containing whitespace, as tokens are separated by whitespace anyway
        ## - ignores examples beginning with #, as these are assumed to be hashtags 
        ## - also ignores examples beginning with @, as these are usertags
        ## - elipses of various lengths are very common so must be handled
        nonword = re.compile('((&(.*);)|[^\s0-9a-zA-Z#@])(\S*)((&(.*);)|[^\s0-9a-zA-Z.])+\S*')
        emoticons = []
        for word in wordBag:
            #print(word)
            if nonword.match(word):
                emoticons.append(word)
        # if not emoticons:
            # print("passing empty list")
        # else:
            # print(emoticons)
        return emoticons
    
    def emoteStrip(self,tokens):
        # print(len(tokens))
        # for token in tokens:
            # print(token)
        postEmotes = [[]]
        for postIndex in range(len(tokens)):
            removals = []
            # token = tokens[postIndex]
            # print("At index " + str(postIndex) + ", there is " + str(tokens[postIndex]))
            postEmotes.insert(postIndex,self.findEmotes(tokens[postIndex]))
            # if len(postEmotes) != 0:
                # print tokens[postIndex]
            tokens[postIndex][:] = [token for token in tokens[postIndex] if token not in postEmotes[postIndex]]
            # for tokenIndex in tokens[postIndex]:
                # print(token not in postEmotes[postIndex])
                # if tokens[postIndex][tokenIndex] in postEmotes:
                    # removals.append[postIndex]
                # if len(postEmotes) != 0:
                    # print tokens[postIndex]
            # if len(postEmotes[postIndex]) != 0:
                # print("The tokens are " + str(tokens[postIndex]))
                # print("The emoticons are " + str(postEmotes[postIndex]))
        return [tokens,postEmotes]
    
    def emoticonTesting(self, mainTest, sentiments, tokens,postEmotes, postProbs):
        ## testing phase
        for postIndex in range(len(sentiments)):
            if mainTest.position[postIndex] == 0:
                predictedIndex = postProbs[postIndex].index(max(postProbs[postIndex]))
                ## Check for emoticons in the post to apply offsets
                if len(postEmotes[postIndex]) != 0:
                    for emote in postEmotes[postIndex]:
                        if self.offsets.has_key(emote):
                            for i in range(3):
                                postProbs[postIndex][i] += self.offsets[emote][predictedIndex][i]
        return postProbs
                                
    def trainEmotes(self,mainTest,sentiments,tokens,postEmotes):
        predicted = [[]]
        postNumber = 0
        differences = {}
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
                    if len(postEmotes[postPosition]) != 0:
                        # print("Predicted place was " + str(predictedIndex) + " with " + str(postProbs[predictedIndex])) 
                        # print("Actual was " + str(actualIndex) + " with " + str(postProbs[actualIndex]))
                        ## Offset for sentence is the difference between the predicted sentiment and the actual, divided by the number of negative sentences
                        for emote in postEmotes[postPosition]:
                            if emote not in differences:
                                differences[emote] = [
                                    [[0.0],[],[]],
                                    [[],[0.0],[]],
                                    [[],[],[0.0]]]
                                differences[emote][actualIndex][predictedIndex].append((postProbs[predictedIndex] - postProbs[actualIndex]) / float(len(postEmotes[postPosition])))
        # print("First column: " + str(differences[0]))
        # print("Second column: " + str(differences[1]))
        # print("Third column: " + str(differences[2]))
        for emote in differences:
            offset = [[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]
            for column in range(3):
                for row in range(3):
                    # print("Column: " + str(column) + ", row: " + str(row))
                    if len(differences[emote][column][row]) != 0:
                        # print(differences[emote][column][row])                        
                        offset[column][row] = sum(differences[emote][column][row]) / float(len(differences[emote][column][row]))
                        # print(offset)
            self.offsets[emote] = offset
                    # else:
                        # self.offsets[emote][column][row] = 0.0
    
    # def emoticonTesting(self,sentiments,tokens):
        # emotePosts = [[]]
        # for posts in range(len(sentiments)):
            # emotePost = self.findEmotes(tokens[posts])
            # print (not emotePost)
            # for emote in emotePost:
                # print(emote)
            # emotePosts.insert(posts,emotePost)
            # if (len(emotePost) != 0): 
                # if position[posts] != 0:
                    # if sentiments[posts] == 4:
                        # print (emotePosts)
                        # posPosts.extend(emotePosts[posts])
                    # elif sentiments[posts] == 0:
                        # negPosts.extend(emotePosts[posts])
                    # else:
                        # neuPosts.extend(emotePosts[posts])

        # emoteBag = defaultdict(list)
        # for token in posPosts:
            # emoteBag[token].append(5)
        # for token in negPosts:
            # emoteBag[token].append(-5)
        # for token in neuPosts:
            # emoteBag[token].append(0)

        # print(emoteBag.keys())

        # emoteSen = {}
        # for emote in emoteBag:
        #    print("Word: " + word + ", instances: " + str(len(wordBag[word])) + ", values: ")
        #    for value in wordBag[word]:
        #            print(value, end=" ")
        #    print("Total = " + str(sum(wordBag[word])))
            # emoteSen[emote] = sum(emoteBag[emote]) / float(len(emoteBag[emote]))
        #    print("word: " + word + ", word sentiment: " + str(wordSen[word]))
            
        ## testing phase
        # corrects = []
        # for posts in range(0,len(sentiments)):
            # if position[posts] == 0:
                # score = 0
                # for token in tokens[posts]:
                    # if token in emoteSen:
                        # score += float(emoteSen[token])
        #                print(score)
                # if score >= 2.5:
                    # corrects.append(sentiments[posts] == 4)
        #            print("postive: " + str(sentiments[posts] == 4))
                # elif score <= -2.5:
                    # corrects.append(sentiments[posts] == 0)
        #            print("negative: " + str(sentiments[posts] == 0))
                # else:
                    # corrects.append(sentiments[posts] == 2)
        #            print("neutral: " + str(sentiments[posts] == 2))
                    
        #print(corrects)
        # numCorrects = sum(corrects)
        # numTestCases = len(corrects)
        # accuracy = float(numCorrects * 100/float(numTestCases))
      # print("The number of correctly predicted posts is " + str(numCorrects) + " out of " + str(numTestCases))
        # return accuracy
