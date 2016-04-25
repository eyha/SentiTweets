from __future__ import print_function
import nltk
#nltk.download()
from nltk import TweetTokenizer
import csv
import random
from collections import defaultdict
from negClauseLearn import negClauseLearner
from bigrams import bigramLearner
from emotIden import emoticonLearner

tests = []
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


class sentiLearn:
    numTestPosts = 0
    priors = []
    wordSen = defaultdict(list)
    evidences = defaultdict(list)
    position = []
    numPosPosts = numNegPosts = numNeuPosts = 0
    
    #Divide into training and test data - randomly allocate 4/5 to training and 1/5 to test
    def getData(self):
        self.position = []
        posPosts = []
        negPosts = []
        neuPosts = []
        for posts in range(0,len(sentiments)):
            self.position.insert(posts,random.randint(0,4))
            if self.position[posts] != 0:
                if sentiments[posts] == 4:
                    posPosts.extend(tokens[posts])
                    self.numPosPosts += 1
                elif sentiments[posts] == 0:
                    negPosts.extend(tokens[posts])
                    self.numNegPosts += 1
                else:
                    neuPosts.extend(tokens[posts])
                    self.numNeuPosts += 1
        self.numTestPosts = self.numPosPosts + self.numNegPosts + self.numNeuPosts
        return [posPosts,negPosts,neuPosts]
    
    def train(self):
        trainData = self.getData()
        #Count the number of instances of each token
        wordBag = defaultdict(list)
        for token in trainData[0]:
            wordBag[token].append(1)
        for token in trainData[1]:
            wordBag[token].append(-1)
        for token in trainData[2]:
            wordBag[token].append(0)
            
        ## Calculate Prior
        self.priors = [0.0,0.0,0.0]
        self.priors[0] = self.numPosPosts / float(self.numTestPosts)
        self.priors[1] = self.numNegPosts / float(self.numTestPosts)
        self.priors[2] = self.numNeuPosts / float(self.numTestPosts)
        # print("Priors are: " + str(self.priors))

        ## Calculate p(x) and likelihoods for words
        self.wordSen = defaultdict(list)
        self.evidences = defaultdict(list)
        for word in wordBag:
            # print("Word: " + word + ", instances: " + str(len(wordBag[word])) + ", values: ")
            # for value in wordBag[word]:
                # print(value, end=" ")
            # print("Total = " + str(sum(wordBag[word])))
            if len(wordBag[word]) > 2:
                # print("For word " + word + ", the length is " + str(len(wordBag[word])) + " and the number of posts is " + str(self.numTestPosts))
                # self.probWordOcc[word] = len(wordBag[word])
                # print(self.probWordOcc[word])
                posEvidence = wordBag[word].count(1) / float(len(wordBag[word])) * self.priors[0] + (wordBag[word].count(-1) + wordBag[word].count(0)) / float(len(wordBag[word])) * (self.priors[1] + self.priors[2])
                negEvidence = wordBag[word].count(-1) / float(len(wordBag[word])) * self.priors[1] + (wordBag[word].count(1) + wordBag[word].count(0)) / float(len(wordBag[word])) * (self.priors[0] + self.priors[2])
                neuEvidence = wordBag[word].count(0) / float(len(wordBag[word])) * self.priors[2] + (wordBag[word].count(-1) + wordBag[word].count(1)) / float(len(wordBag[word])) * (self.priors[1] + self.priors[1])
                self.evidences[word].extend([posEvidence,negEvidence,neuEvidence])            
                
                posLike = wordBag[word].count(1) / float(len(wordBag[word]))
                negLike = wordBag[word].count(-1) / float(len(wordBag[word]))
                neuLike = wordBag[word].count(0) / float(len(wordBag[word]))
                self.wordSen[word] = [posLike,negLike,neuLike]
                # print("word: '" + word + "', word sentiment: " + str(self.wordSen[word]))

    def postCheck(self,postIndex):
        postProbs = [1.0,1.0,1.0]
        for i in range(3):
            for token in tokens[postIndex]:
                if token in self.wordSen:
                    if self.wordSen[token][i] != 0.0:
                        postProbs[i] *= (self.wordSen[token][i])
                        # print("Position " + str(i) + " - postsProb after token '" + token + "': " + str(postProbs[i]))
                    if self.evidences[token][i] != 0:
                        # evidence = self.wordSen[token][i] * self.priors[i]
                        # evidence += ((sum(self.wordSen[token]) - self.wordSen[token][i]) * (sum(self.priors) - self.priors[i]))
                        postProbs[i] /= float(self.evidences[token][i])
                    # print("Position " + str(i) + " - evidence after token '" + token + "': " + str(self.evidences[token][i]))
                    # print(self.probWordOcc[token])
                    # print(postProbs)
            postProbs[i] *= float(self.priors[i])
        # print("Final Probability is: " + str(postProbs))
        return postProbs

    def test(self):    
        #testing phase
        corrects = []
        for postIndex in range(len(sentiments)):
            if self.position[postIndex] == 0:
                postProbs = self.postCheck(postIndex)
                predictedIndex = postProbs.index(max(postProbs))
                if predictedIndex == 0:
                    corrects.append(sentiments[postIndex] == 4)
                    # print("postive: " + str(sentiments[postIndex] == 4))
                elif predictedIndex == 1:
                    corrects.append(sentiments[postIndex] == 0)
                    # print("negative: " + str(sentiments[postIndex] == 0))
                else:
                    corrects.append(sentiments[postIndex] == 2)
                    # print("neutral: " + str(sentiments[postIndex] == 2))
        ##print(corrects)
        numCorrects = sum(corrects)
        numTests = len(corrects)
        accuracy = float(numCorrects*100/float(numTests))
        # tests.append(accuracy)
        # print("The number of correctly predicted posts is " + str(numCorrects) + " out of " + str(numTests) + ".")
        # print("The accuracy was " + str(accuracy) + "%")
        return accuracy

output = open('SentiTweetsOutput.csv', 'ab')
writer = csv.writer(output, 'excel')

sentTest = sentiLearn()    
for t in range(0,5):
    sentTest.train()
    tests.append(sentTest.test())
print("Standard test accuracy: " + str(tests) + ", avg: " + str(sum(tests) / float(len(tests))))
writer.writerow(tests)
output.close()

# Checks for negations in sentences and creates a list of affected words
tests = []
output = open('negClauseLearnOutput.csv', 'ab')
writer = csv.writer(output, 'excel')
negClauseTest = negClauseLearner()
for t in range(0,5):
    sentTest.train()
    negClauseTest.trainNeg(sentTest,sentiments,tokens)
    tests.append(negClauseTest.negTesting(sentTest,sentiments,tokens))
print("Negation test accuracy: " + str(tests) + ", avg: " + str(sum(tests) / float(len(tests))))
writer.writerow(tests)
output.close()

tests = []
output = open('emotIdenOutput.csv', 'ab')
writer = csv.writer(output, 'excel')
emotIdenTest = emoticonLearner()
for t in range(0,5):
    splitted = emotIdenTest.emoteStrip(tokens)
    tokens = splitted[0]
    sentTest.train()
    emotIdenTest.trainEmotes(sentTest,sentiments,tokens,splitted[1])
    tests.append(emotIdenTest.emoticonTesting(sentTest,sentiments,tokens,splitted[1]))
print("Emoticon accuracy: " + str(tests) + ", avg: " + str(sum(tests) / float(len(tests))))
writer.writerow(tests)
output.close()

tests = []
# output = open('emotIdenOutput.csv', 'ab')
# writer = csv.writer(output, 'excel')
# emotIdenTest = emoticonLearner()
for t in range(0,5):
    splitted = emotIdenTest.emoteStrip(tokens)
    tokens = splitted[0]
    sentTest.train()
    negClauseTest.trainNeg(sentTest,sentiments,tokens)
    emotIdenTest.trainEmotes(sentTest,sentiments,tokens,splitted[1])
    tests.append(emotIdenTest.emoticonTesting(sentTest,sentiments,tokens,splitted[1]))
print("Combined accuracy: " + str(tests) + ", avg: " + str(sum(tests) / float(len(tests))))
# writer.writerow(tests)
# output.close()



tests = []
output = open('bigramOutput.csv', 'ab')
writer = csv.writer(output, 'excel')
bigramTest = bigramLearner()
for t in range(0,5):
    tests.append(bigramTest.bigramTesting(sentiments,tokens))
print(tests)
writer.writerow(tests)
output.close()
