# To deal with folders directories and files pathes and names
from os.path import normpath, basename
from glob import glob

# To deal with texts
import re
from collections import Counter
from itertools import chain

# Data structure used in this assignement and plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Bonus part
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, cross_val_predict
from sklearn import svm
from sklearn import metrics

# DO NOT CHANGE THE SIGNATURES OF ANY DEFINED FUNCTIONS.
# YOU CAN ADD "HELPER" FUNCTIONS IF YOU LIKE.

def part1_load(folder1, folder2, n=1):
    '''
    Read text files under folder1 and folder2 and return words count in pandas dataframe
    '''
    # Initialize datafaramme to hold Words Count for all files in all folders
    allWrdsCount = pd.DataFrame()
    pattern = r"[!\"#$%\\\'()*+,\-./:;<=>?@\[\\\]\^_`{\|}~]"     # Punctuations to remove
    regex = re.compile(pattern)
    
    # Count number of words in each file in each folder
    for folder in [folder1, folder2]:
        myclass = basename(normpath(folder))
        allFiles = glob("{}/*.txt".format(folder))
        
        for myfile in allFiles:
            with open(myfile) as f:
                # word preparation
                #txtPrepared = CleanText(f, pattern)
                #wrdsCount = WordsCounter(txtPrepared)
                txtClean = ([re.sub(regex, '', line) for line in f])
                txtClean = map(str.lower, txtClean)
                txtClean = map(str.split, txtClean)
                wrdsCount = Counter(chain.from_iterable(txtClean))

            #wrdsCount["Class"] = myclass
            #wrdsCount["File_Name"] = myfilename
            #Construct DF for currunt file
            myfilename = basename(myfile)
            idx = pd.MultiIndex.from_arrays([[myclass], [myfilename]], names=['Class', 'Filename'])
            fileWrdCount = pd.DataFrame(wrdsCount, index=idx)
            
            allWrdsCount = allWrdsCount.append(fileWrdCount, ignore_index=False)

    allWrdsCount.fillna(0, inplace=True)    # convert NaN to zeros
    allWrdsCount.astype(np.uint32).dtypes
    
    # Filter those counrs < n
    columnTotals = allWrdsCount.sum()
    columnTotalGeN = columnTotals.ge(n)
    columnNames = columnTotalGeN[columnTotalGeN == True].index
    allWrdsCount = allWrdsCount.filter(items=columnNames)
    #allWrdsCount.to_csv('out.csv')
    return allWrdsCount

def part2_vis(df, m):
    # DO NOT CHANGE
    assert isinstance(df, pd.DataFrame)

    if df.empty:
        return None
    else:
        # Get a list of m words that have the max occurances
        columnTotals = df.sum()
        colsMaxm = columnTotals.nlargest(m, keep='all') # Keep duplication   
        maxMWords = colsMaxm.index.to_numpy()
        # Construct dataframe that holds the count of occurances for
        # the most frequent m words that have for each class 
        #maxMWords_df = df[maxMWords].sum(level=0)
        maxMWords_df = df.filter(items=maxMWords)
        maxMWords_df = maxMWords_df.reindex(columns=maxMWords)
        maxMWords_df = maxMWords_df.sum(level=0)
        #columnTotals.to_csv('TotalCount.csv')
        return maxMWords_df.T.plot(kind="bar")

def part3_tfidf(df):
    # DO NOT CHANGE
    assert isinstance(df, pd.DataFrame)
    N = df.shape[0] # number of documents
    if N > 0:
        nt = (df > 0).sum()                     # number of documents where the term appear
        idf = np.log( N / (0.0001 + nt) )       # add 0.0001 to the count to avoide division by zero. 
                                                # I don't like adding 1, I am not used to it :) but it is named by Laplace!
        tf_idf = df.mul(idf)
        return tf_idf
    else:
        return None

def bonus_classifier(df):
    # Construxt the features matrix "X" and the target matrix "y"
    X = df.reset_index(drop=True)
    y = df.index.get_level_values(0).to_series(index=np.arange(X.shape[0]))
    # Use Linear SVM and k-fold cross validation in classification.
    clf = {"model": [], "scores": [], "confusion_matrix": [], "predict": []}
    CV = StratifiedKFold(n_splits=5)
    #CV = KFold(n_splits=5)
    for trainIndex, testIndex in CV.split(X, y):
        XTrain, XTest = X.iloc[trainIndex], X.iloc[testIndex]
        yTrain, yTest = y.iloc[trainIndex], y.iloc[testIndex]

        model = svm.SVC(kernel='linear')
        model.fit(XTrain, yTrain)
        yPredict = model.predict(XTest)
        score = metrics.accuracy_score(yTest, yPredict)
        confMatrix = metrics.confusion_matrix(yTest, yPredict)
        
        clf["model"].append(model)
        clf["predict"].append(yPredict)
        clf["scores"].append(score)
        clf["confusion_matrix"].append(confMatrix)
    
    #scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    #y_pred = cross_val_predict(clf, X, y, cv=5)
    #scoresMean = scores.mean()
    #scroresSTD = scores.std() * 2
    return clf #[scoresMean, scroresSTD]
    

# ADD WHATEVER YOU NEED HERE, INCLUDING BONUS CODE.
def CleanText(text, regex):
    ''' Remove punctuation, lower case and split each line into a list
    This generator function takes TextIOWrapper and a compiled regex pattern'''
    for line in text:
        txt = re.sub(regex, '', line)
        txt = txt.lower()
        txt = txt.split()
        yield txt

def WordsCounter(text):
    countDict = {}
    for word in chain.from_iterable(text):
        if word in countDict: 
            countDict[word] +=1
        else: 
            countDict[word] =1
    return countDict

if __name__ == "__main__":
    class1 = "../../Corpora/reuters-topics/grain"
    class2 = "../../Corpora/reuters-topics/crude"
    counts = part1_load(class1, class2, 100)
    plot = part2_vis(counts,10)
    bonus_classifier(counts)
    pass
