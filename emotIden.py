from __future__ import print_function
import nltk
#nltk.download()
from nltk import TweetTokenizer
import csv
import random
from collections import defaultdict
import re

#tokenizer = TweetTokenizer()
#    csvfile = open('trainingandtestdata/testdata.manual.2009.06.14.csv', 'rb')
#    reader = csv.reader(csvfile, delimiter=',')
#    rownum = 0
#    sentiments = []
#    tokens = [[]]
#    for row in reader:
#        colnum = 0
#        for col in row:
#            if colnum == 0:
#                sentiments.insert(rownum,int(col))
#            if colnum == 5:
#                raw = col #.read().decode('utf8')
#                tokens.insert(rownum,tokenizer.tokenize(raw))
    ##            print("tokens contents:", end='')
    ##            for word in tokens[rownum]:
    ##                print(word, end = " ")
    ##            print()
#            colnum += 1
#        rownum += 1
#    csvfile.close()

class emoticonLearner:
    def findEmotes(self,wordBag):
        # match all tokens that have more than one special characters - need to handle HTML special characters
        # - can't handle ones containing whitespace, as tokens are separated by whitespace anyway
        # - ignores examples beginning with #, as these are assumed to be hashtags 
        # - also ignores examples beginning with @, as these are usertags
        # - elipses of various lengths are very common so must be handled
        nonword = re.compile('((&(.*);)|[^\s0-9a-zA-Z#@])(\S*)((&(.*);)|[^\s0-9a-zA-Z.])+\S*')
        emoticons = []
        for word in wordBag:
            #print(word)
            if nonword.match(word):
                emoticons.append(word)
        #if not emoticons:
        #    print("passing empty list")
        #else:
        #    print(emoticons)
        return emoticons

    def emoticonTesting(self,sentiments,tokens):
        #Divide into training and test data - randomly allocate 4/5 to training and 1/5 to test
        position = []
        posPosts = []
        negPosts = []
        neuPosts = []
        emotePosts = [[]]
        for posts in range(0,len(sentiments)):
            position.insert(posts,random.randint(0,4))
            emotePost = self.findEmotes(tokens[posts])
            #print (not emotePost)
            #for emote in emotePost:
            #    print(emote)
            emotePosts.insert(posts,emotePost)
            if (len(emotePost) != 0): 
                if position[posts] != 0:
                    if sentiments[posts] == 4:
                        #print (emotePosts)
                        posPosts.extend(emotePosts[posts])
                    elif sentiments[posts] == 0:
                        negPosts.extend(emotePosts[posts])
                    else:
                        neuPosts.extend(emotePosts[posts])

        emoteBag = defaultdict(list)
        for token in posPosts:
            emoteBag[token].append(5)
        for token in negPosts:
            emoteBag[token].append(-5)
        for token in neuPosts:
            emoteBag[token].append(0)

        # print(emoteBag.keys())

        emoteSen = {}
        for emote in emoteBag:
        ##    print("Word: " + word + ", instances: " + str(len(wordBag[word])) + ", values: ")
        ##    for value in wordBag[word]:
        ##            print(value, end=" ")
        ##    print("Total = " + str(sum(wordBag[word])))
            emoteSen[emote] = sum(emoteBag[emote]) / float(len(emoteBag[emote]))
        ##    print("word: " + word + ", word sentiment: " + str(wordSen[word]))
            
        #testing phase
        corrects = []
        for posts in range(0,len(sentiments)):
            if position[posts] == 0:
                score = 0
                for token in tokens[posts]:
                    if token in emoteSen:
                        score += float(emoteSen[token])
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
#       print("The number of correctly predicted posts is " + str(numCorrects) + " out of " + str(numTestCases))
        return accuracy
