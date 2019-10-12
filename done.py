# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:53:18 2019

@author: Bivash
@ID: 1001558655
"""


import os
import math
import operator
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpusroot = './presidential_debates'
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
sword = stopwords.words('english')
stemmer = PorterStemmer()

totalDocuments = 0 # total document count
document = {} # dictionary of filename and tokens
factor = {} # dictionary of filename and normalizing lengths
postinglist = {} # list of all tokens and their corresponding file names in descending order
globaltokenlist = set() # Unique set of tokens

# Reading the files, removing stopwords, and stemming
for filename in os.listdir(corpusroot):
    totalDocuments += 1
    file = open(os.path.join(corpusroot, filename), "r", encoding='UTF-8')
    doc = file.read()
    file.close()
    doc = doc.lower()
    tokens = tokenizer.tokenize(doc)
    tokens = [stemmer.stem(token) for token in tokens if token not in sword]
    globaltokenlist.update(tokens)
    tf = dict(Counter(tokens))
    document[filename] = tf

# Get the idf score of a given token
def getidf(token):
    df = 0
    for filename in document:
        temp = document[filename]
        if token in temp:
            df += 1
    if df == 0:
        return -1    
    
    return math.log10(totalDocuments / df)

# Get the raw tf-idf score without normalizing
def getrawweight(filename, token):
    if token not in document[filename]:
        return 0

    tf = 1 + math.log10( document[filename][token] )
    idf = getidf(token)
    score = tf * idf 
    return score

# Get the normalized tf-idf score
def getweight(filename, token):
    if token not in document[filename]:
        return 0
    return document[filename][token]

# Function to calculate the tf-idf score
def calculatescore():
    for filename in document:
        length = 0
        for token in document[filename]:
            score = getrawweight(filename, token)
            document[filename][token] = score
            length += score ** 2
        
        length = math.sqrt(length)
        factor[filename] = length

# Function to normalize tf-idf scores
def getnormalizedscore():
    for filename in document:
        temp = factor[filename]
        for token in document[filename]:
            document[filename][token] = document[filename][token] / temp

# Function to create the posting lists of all the unique tokens
def createpostingslist():
    for token in globaltokenlist:
        tempdict = {}
        for filename in document:
            if token in document[filename]:
                tempdict[filename] = float(document[filename][token])
        tempdict = sorted(tempdict.items(), key = operator.itemgetter(1), reverse = True)
        postinglist[token] = tempdict
    
# Function to return the cosine similarity of given query in the form of a tuple
# of (filename, score)
def query(qstring):
    qstring = qstring.lower()
    qstring = tokenizer.tokenize(qstring)
    tokens = [stemmer.stem(token) for token in qstring if token not in sword]
    tokens = dict(Counter(tokens))
    normalizingfactor = 0
    
    # tf weight
    for token in tokens:
        tokens[token] = 1 + math.log10(tokens[token])
        normalizingfactor += tokens[token] ** 2

    normalizingfactor = math.sqrt(normalizingfactor)
    
    present = "no"
    # normalizing tf score
    for token in tokens:
        tokens[token] = tokens[token] / normalizingfactor
        if token in postinglist:
            present = "yes"
    
    # Check if no tokens from query in corpus
    if present == "no":
        return ("None", 0.000000000000)
    
    documentsim = {} #  dict of filename and cosine similarity
    notintopten = [] # not in top ten list to see if we need to fetch more
    toptenpostinglist = {} # list of top ten files with tokens
    
    # loop to insert 10 values into toptenpostinglist
    for token in tokens:
        if token in postinglist:
           temp = dict(postinglist[token][0:10])
           toptenpostinglist[token] = temp
    
    # initializing uniques files into documentsim dictionary
    for token in toptenpostinglist:
        for filename in toptenpostinglist[token]:
            if filename not in documentsim:
                documentsim[filename] = 0
            
    # Loop for finding cosine similarity
    for token in toptenpostinglist:
        for filenames in documentsim:
            if filenames in toptenpostinglist[token]:
                documentsim[filenames] += toptenpostinglist[token][filenames] * tokens[token]
            else:
                # Else use upper bound
                documentsim[filenames] += postinglist[token][9][1] * tokens[token]
                notintopten.append(filenames)    
                
    documentsim = sorted(documentsim.items(), key = operator.itemgetter(1), reverse = True)
    
    if (documentsim[0][0] in notintopten):
        return ("fetch more", 0.0)
    else:
        return (documentsim[0][0],documentsim[0][1])            
   
# Calling all relevant functions    
calculatescore()
getnormalizedscore()
createpostingslist()


# Test Cases
print("(%s, %.12f)" % query("health insurance wall street"))
print("(%s, %.12f)" % query("security conference ambassador"))
print("(%s, %.12f)" % query("particular constitutional amendment"))
print("(%s, %.12f)" % query("terror attack"))
print("(%s, %.12f)" % query("vector entropy"))
print("(%s, %.12f)" % query("health insurance wall street"))

print("%.12f" % getweight("2012-10-03.txt","health"))
print("%.12f" % getweight("1960-10-21.txt","reason"))
print("%.12f" % getweight("1976-10-22.txt","agenda"))
print("%.12f" % getweight("2012-10-16.txt","hispan"))
print("%.12f" % getweight("2012-10-16.txt","hispanic"))
print("%.12f" % getidf("health"))
print("%.12f" % getidf("agenda"))
print("%.12f" % getidf("vector"))
print("%.12f" % getidf("reason"))
print("%.12f" % getidf("hispan"))
print("%.12f" % getidf("hispanic"))
print("(%s, %.12f)" % query("terror attacks Aniket"))
print("(%s, %.12f)" % query("some more terror attack"))
print("%.12f" % getweight('1960-10-13.txt','identifi'))
print("%.12f" % getweight('1976-10-22.txt','institut'))
print("%.12f" % getidf('andropov'))
print("%.12f" % getidf('identifi'))


