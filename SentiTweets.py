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
    # probWordOcc = {}
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

        # Calculate p(x) and likelihoods for words
        self.wordSen = defaultdict(list)
        # self.probWordOcc = {}
        self.evidences = defaultdict(list)
        for word in wordBag:
        ##    print("Word: " + word + ", instances: " + str(len(wordBag[word])) + ", values: ")
        ##    for value in wordBag[word]:
        ##            print(value, end=" ")
        ##    print("Total = " + str(sum(wordBag[word])))
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

    def test(self):    
        #testing phase
        corrects = []
        for posts in range(len(sentiments)):
            if self.position[posts] == 0:
                postProbs = [1,1,1]
                for i in range(3):
                    for token in tokens[posts]:
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
                predictedIndex = postProbs.index(max(postProbs))
                if predictedIndex == 0:
                    corrects.append(sentiments[posts] == 4)
                    # print("postive: " + str(sentiments[posts] == 4))
                elif predictedIndex == 1:
                    corrects.append(sentiments[posts] == 0)
                    # print("negative: " + str(sentiments[posts] == 0))
                else:
                    corrects.append(sentiments[posts] == 2)
                    # print("neutral: " + str(sentiments[posts] == 2))

        ##print(corrects)
        numCorrects = sum(corrects)
        numTests = len(corrects)
        accuracy = float(numCorrects*100/float(numTests))
        tests.append(accuracy)
        # print("The number of correctly predicted posts is " + str(numCorrects) + " out of " + str(numTests) + ".")
        # print("The accuracy was " + str(accuracy) + "%")
        return accuracy    

# Checks for negations in sentences and creates a list of affected words
# def negCheck(tokens):
    # positives = []
    # negatives = []
    # currentSentence = []
    # negative = False
    # for token in tokens:
        # if(token in negation):
            # negative = true
        # elif(re.match('[.!?]+')):
            # if negative:
                # negatives.extend(currentSentence)
                # currentSentence = []
                # negative = false
            # else:
                # positives.extend(currentSentence)
                # currentSentence = []
    ##if not currentSentence:
    ##   break
    # if negative:
        # negatives.extend(currentSentence)
        # currentSentence = []
        # negative = false
    # else:
        # positives.extend(currentSentence)
        # currentSentence = []
    # return (positives,negatives)

# def negTesting():
    # Divide into training and test data - randomly allocate 4/5 to training and 1/5 to test
    # position = []
    # posPosts = []
    # negPosts = []
    # neuPosts = []
    # for posts in range(0,len(sentiments)):
        # position.insert(posts,random.randint(0,4))
        # if position[posts] != 0:
            # posNegs = negCheck(tokens[posts])
            # if sentiments[posts] == 4:
                # posPosts.extend(posNegs[0])
                # negPosts.extend(posNegs[1])
            # elif sentiments[posts] == 0:
                # negPosts.extend(posNegs[1])
                # posPosts.extend(posNegs[0])
            # else:
                # neuPosts.extend(posNegs[0])
                # neuPosts.extend(posNegs[1])

    # wordBag = defaultdict(list)
    # for token in posPosts:
        # wordBag[token].append(5)
    # for token in negPosts:
        # wordBag[token].append(-5)

    # for token in neuPosts:
        # wordBag[token].append(0)

    # wordSen = {}
    # for word in wordBag:
    #    print("Word: " + word + ", instances: " + str(len(wordBag[word])) + ", values: ")
    #    for value in wordBag[word]:
    #            print(value, end=" ")
    #    print("Total = " + str(sum(wordBag[word])))
        # wordSen[word] = sum(wordBag[word]) / float(len(wordBag[word]))
    #    print("word: " + word + ", word sentiment: " + str(wordSen[word]))
        
    #testing phase
    # corrects = []
    # for posts in range(0,len(sentiments)):
        # if position[posts] == 0:
            # score = 0
            # for token in tokens[posts]:
                # if token in wordSen:
                    # score += float(wordSen[token])
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
    # numTests = len(corrects)
    # accuracy = float(numCorrects*100/float(numTests))
    # tests.append(accuracy)
    # print("The number of correctly predicted posts is " + str(numCorrects) + " out of " + str(numTests) + ".")
    # print("The accuracy was " + str(accuracy) + "%")
    # return accuracy

output = open('SentiTweetsOutput.csv', 'ab')
writer = csv.writer(output, 'excel')

sentTest = sentiLearn()    
for t in range(0,5):
    sentTest.train()
    tests.append(sentTest.test())
print(tests)
writer.writerow(tests)
output.close()

tests = []
output = open('negClauseLearnOutput.csv', 'ab')
writer = csv.writer(output, 'excel')
negClauseTest = negClauseLearner()
negClauseTest.trainNeg(sentTest,sentiments,tokens)
for t in range(0,5):
    tests.append(negClauseTest.negTesting(sentiments,tokens))
print(tests)
writer.writerow(tests)
output.close()

tests = []
output = open('emotIdenOutput.csv', 'ab')
writer = csv.writer(output, 'excel')
emotIdenTest = emoticonLearner()
for t in range(0,5):
    tests.append(emotIdenTest.emoticonTesting(sentiments,tokens))
print(tests)
writer.writerow(tests)
output.close()

tests = []
output = open('bigramOutput.csv', 'ab')
writer = csv.writer(output, 'excel')
bigramTest = bigramLearner()
for t in range(0,5):
    tests.append(bigramTest.bigramTesting(sentiments,tokens))
print(tests)
writer.writerow(tests)
output.close()
