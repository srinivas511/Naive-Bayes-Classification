import sys
import re 

def getStopWords(fileName):
    pf = open (fileName, 'r')
    stopWordsList = []
    for line in pf:
        line = line.strip()
        stopWordsList.append(line)
    pf.close()    
    return stopWordsList

def getStringList(fileName):
    pf = open(fileName, 'r',encoding='utf-8', errors='ignore')
    strList = [] #string list for storing each line(a string)
    line = pf.readline().lower() #convert str to lower case
    while (line != ""): #read to the EOF, return an empty line
        strList.append(line)
        line = pf.readline().lower()
    pf.close()

    content = ""
    content = ''.join(strList) #construct string list, then join them
    return content

def getwordslist(fileContent):
    expr = '\s+| |\W+' 
    wordList = []
    wordList = re.split(expr, fileContent)
    wordList = [item for item in wordList if item.isalpha()]
    return wordList

def removeStopWords(fileWordList, stopWordList):
    fileWordList = filter(lambda item: item not in stopWordList, fileWordList)
    return fileWordList

def getAllWords(docName, stopWordFileName):
    wordswithSW = getwordslist(getStringList(docName))    
    stopWords = getStopWords(stopWordFileName)
    wordsList = removeStopWords(wordswithSW, stopWords)
    return wordsList