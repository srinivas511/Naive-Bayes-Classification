# -*-coding:utf-8 -*-
import sys
import os
import numpy as np
from preprocess import getAllWords
import pickle as p
import csv
import math
from datetime import datetime

global stopWordFileName
stopWordFileName = r"stoplist.txt"
global parentDir
parentDir = "articles"

def getVocabulary(rootDir):
    fileList = [] 
    for parent, dirNames, fileNames in os.walk(rootDir):        
        for fileName in fileNames:
            fileList.append(os.path.join(parent, fileName))

    tempVocab = []
    for eachfile in fileList:
        wordsList = getAllWords(eachfile, stopWordFileName)
        tempVocab += wordsList
    tempVocab.sort()

    vocabCountDict = {}
    for word in tempVocab:
        if word not in vocabCountDict:
            vocabCountDict[word] = 1 
        else:
            vocabCountDict[word] += 1

    Vocabulary = vocabCountDict.keys()
    
    vocabFile = "vocabulary.txt"
    w = csv.writer(open('vocabulary.csv', 'w',errors='ignore'))
    for key in vocabCountDict.keys():
        w.writerow([key,vocabCountDict[key]])
    f = open(vocabFile, 'wb')
    p.dump(vocabCountDict, f) #dump the object into a file
    f.close()

    return Vocabulary
	
def ClassNames(rootDir):
    clsNames = []
    clsNames = os.listdir(rootDir)
    clsNames.sort()
    return clsNames

def ClassProbability(rootDir,numofclasses):
    clsList = []
    clsList = ClassNames(rootDir)

    clsPriorProb = [[(x+1.0/numofclasses)] for x in np.zeros(numofclasses, np.float)]

    clsPriorProbDict = {}
    for i in range( len(clsList) ):
        clsPriorProbDict[clsList[i]] = clsPriorProb[i]

    clsPriorProbFile = "clsPriorProb.txt"
    w = csv.writer(open('clsPriorProb.csv', 'w'))
    for key in clsPriorProbDict.keys():
        w.writerow([key,clsPriorProbDict[key]])
    f = open(clsPriorProbFile, 'wb')
    p.dump(clsPriorProbDict, f)
    f.close()
    
    return clsPriorProbDict 	

def MergeClassFiles(rootDir):
    clsList = ClassNames(rootDir) 
        
    dirList = []
    for parent, dirNames, fileNames in os.walk(rootDir):        
        for dirName in dirNames:
            dirList.append(os.path.join(parent, dirName))
    dirList.sort()

    clsWordDict = {}
    
    for i in range(len(clsList)):
        fileList = []
        for parent, dirNames, fileNames in os.walk(dirList[i]):        
            for fileName in fileNames:
                fileList.append(os.path.join(parent, fileName))

        #combine the files of a class into clsWordsList, each file pre-processed
        trainSize = int(len(fileList)*0.5) #50% of the documents for training
        clsWordsList = []
        for j in range(trainSize):
            clsWordsList += getAllWords(fileList[j], stopWordFileName)
        clsWordDict[clsList[i]] = clsWordsList

    return clsWordDict

def ClassWordProb(vocab, clsWordDict):
    vocabSize = len(vocab)
    vocablist = list(vocab)
    ClassWordProbDict = {}  
    for className in list(clsWordDict):
        clsWordsList = clsWordDict[className]
        docWordCount = len(clsWordsList)

        ClassWordProbList = []
        for j in range(vocabSize):
            occurences = clsWordsList.count(vocablist[j])
            ClassWordProb = (occurences + 1.0)/(docWordCount + vocabSize)
            ClassWordProbList.append(ClassWordProb)

        ClassWordProbDict[className] = ClassWordProbList
    
    ClassWordProbFile = "ClassWordProb.txt"
    f = open(ClassWordProbFile,'wb')
    p.dump(ClassWordProbDict,f)
    f.close()
    return ClassWordProbDict

def getTextFeature(docName, vocabulary):
    textWordsList = []
    textWordsList = getAllWords(docName, stopWordFileName)
    textWordsList = sorted(textWordsList)
    textFeatureList = []
    for eachWord in vocabulary:
        textFeatureList.append(textWordsList.count(eachWord))

    return textFeatureList

def classifyText(textFeatureList, clsPriorProbDict, ClassWordProbDict):
    textFeatureList = [1,] + textFeatureList #X=(1, textFeatureList)
    textFeatureVector = np.array(textFeatureList)

    weightProbDict = {} 
    for eachClass in clsPriorProbDict.keys():
        weightProbDict[eachClass] = clsPriorProbDict[eachClass]+ClassWordProbDict[eachClass]
        
    clsConfidenceDict = {}

    for eachClass in list(weightProbDict):
        weightProbList = weightProbDict[eachClass]
        weightProbList = [np.float(math.log(item)) for item in weightProbList]
        weightProbVector = np.array(weightProbList)
        
        clsConfidenceDict[eachClass] = np.dot(weightProbVector,textFeatureVector)
    clsTag=max(clsConfidenceDict.keys(),key=lambda k:clsConfidenceDict[k])
    return clsTag 

def NaiveBayesAccuracy(rootDir, clsPriorProbDict, ClassWordProbDict, vocabulary):
    f = open('Classificationresults.txt','w')
    clsDirDict = {}
    for parent, dirNames, fileNames in os.walk(rootDir):        
        for dirName in dirNames:
            clsDirDict[dirName] = os.path.join(parent, dirName)

    global totalErrorCount  #error classified document's total count
    global totalFileCount    #total document count
    totalFileCount = 0
    totalErrorCount = 0
    
    textFeatureList = []
    
    for clsName, clsDir in clsDirDict.items():
        actualclass = clsName
        classfileerrcount = 0
        fileList = [] 
        for parent, dirNames, fileNames in os.walk(clsDir):        
            for fileName in fileNames:
                fileList.append(os.path.join(parent, fileName))
        
        clsFileCount = len(fileList)
        testSize = int(0.5*clsFileCount)
        totalFileCount += testSize #total document count increase class by class        

        for j in range( (clsFileCount-testSize) , clsFileCount):
            textFeatureList = getTextFeature(fileList[j], vocabulary)
            predclass = classifyText(textFeatureList, clsPriorProbDict, ClassWordProbDict)
            f.write('-------------------------')
            f.write('\n'+str(fileList[j])+'\n')
            f.write("Actual Class: "+actualclass+'\n')
            f.write("Predicted Class: "+predclass+'\n')
            f.write('-------------------------\n')
            if predclass != actualclass:
                classfileerrcount+=1
                totalErrorCount += 1
        print("Accuracy of class",clsName," is: ",(1-1.0*classfileerrcount/testSize)*100)
    f.close()
    return 1.0*totalErrorCount/totalFileCount 

if __name__ == "__main__":
    start = datetime.now()

    vocabularyList = getVocabulary(parentDir)
    vocabFile = "vocabulary.txt"
    vocabulary = []
    vocabulary = p.load(open(vocabFile, 'rb'))  

    clsList = []
    clsList = ClassNames(parentDir)
	
    clsPriorProbDict = ClassProbability(parentDir,len(clsList))

    clsWordDict = MergeClassFiles(parentDir)
    
    ClassWordProbDict = ClassWordProb(vocabulary, clsWordDict)
    for clsName in list(ClassWordProbDict):
        print(clsName, len(ClassWordProbDict[clsName]))
    errorRate = NaiveBayesAccuracy(parentDir, clsPriorProbDict, ClassWordProbDict, vocabulary)
    print("Naive Bayes Classifier's Accuracy is :",(1 - errorRate)*100)
    print("Running Time of the program is:",(datetime.now()-start))