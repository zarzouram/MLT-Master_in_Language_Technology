import argparse
import random
from sklearn.datasets import fetch_20newsgroups
from sklearn.base import is_classifier
import numpy as np
import pandas as pd

# To deal with texts
import re
from collections import Counter
from itertools import chain

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
#from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score

random.seed(42)
# Print some radom data for evaluation
# randomvalues=[]
# for _ in range(20):
#     randomvalues.append(random.randint(0, 18846))
# path="/home/guszarzmo@GU.GU.SE/LT2212-v20/a2/texts/"

import re
# RE to match headers
patternhdr = r"(^.*?(^>? ?In )((article <[\w\.-]+@[\w\.-]+>)|(<[\w\.-]+@[\w\.-]+>)).*?$)|(^.*?((([\w\.-]+@[\w\.-]+ )?(\(\w+ \w+\) ))(write|wrote|says)).*?$)|^>(<?[\w\.-]+@[\w\.-]+>?)?$|((^.*?^From:.*?$)|(^.*?^Subject:.*?$)|(^.*?^Organization:.*?$)|(^.*?^Lines:.*?$)|(^.*?^To:.*?$)|(^.*?^CC:.*?$)|(^.*?-Posting-Host:.*?$)|(^.*?^Summary:.*?$)|(^.*?^Reply-To:.*?$)|(^.*?^Article-I\.D\.:.*?$)|(^.*?Distribution:.*?$)|(^.*?Newsreader:.*?$))"
# RE to match headers
patternsignature = r"(^(.|(\w*\s)){0,2}[r|R]egard(s?),?$)|(^ ?((-|=|\*|\+|/|\\) ?){7,}|^ ?(- ?){2,}$)"

pattern_spesific = r'''(?x)@[\w\.-]+|(?:\d{1,2}:\d{1,2}(?:[ ]?PM|[ ]?AM|[ ]?pm|[ ]?am)?)|\d+(?:\.\d+)?[%|$]|^\$\d+(?:\.\d+)?|\b(?:\w+)(?:'|-|_)\w+(?=[\s.,?!])'''
pattern_specialccsatb = r'''[!\"#$%\\\'()*+,\-./:;<=>?@\[\\\]\^_`{\|}~]+(?=\W)|(?<=\W)[!\"#$%\\\'()*+,\-./:;<=>?@\[\\\]\^_`{\|}~]+|/|\\'''
alpha = r'''(?<=[\s])[a-z]+(?=[\s])'''

regexhdr = re.compile(patternhdr, re.M)
regexsignature = re.compile(patternsignature, re.M)
regexpc = re.compile(pattern_spesific, re.M)
regexps = re.compile(pattern_specialccsatb, re.M)
regexabc = re.compile(alpha, re.M)


###### PART 1
#DONT CHANGE THIS FUNCTION
def part1(samples):
    #extract features
    X = extract_features(samples)
    assert type(X) == np.ndarray
    print("Example sample feature vec: ", X[0])
    print("Data shape: ", X.shape)
    return X


def extract_features(samples):
    print("Extracting features ...")
    textsclean = clean_text(samples)    # remove header and signature
    textsprepared = prepare_text(textsclean)    # lower word case
    tokenzpertext = tokenize_text(textsprepared) # split text into tokens

    # counting words in each file
    tokenzcount = list(count_words(tokenzpertext))
    # all tokez in dataset
    tokenzuniqglobal = list(set(chain.from_iterable(tokenzcount))) 
    tokenzuniqglobal.sort()
    # map each token to its position in the feature matrix
    tokenzuniqidmap = {t: id for id,t in enumerate(tokenzuniqglobal)}
    
    # initialize feature samples
    nrows = len(samples); ncols = len(tokenzuniqglobal)
    features = np.zeros((nrows, ncols), dtype=np.uint16)
    # fill features matrix
    for i in range(nrows):
        tokenz = tokenzcount[i].keys()
        for token in tokenz:
            colid = tokenzuniqidmap[token]
            features[i, colid] = tokenzcount[i][token]
    
    # filter those words that are more than 10 (number of labels/2)
    tokenztotalcounts = np.sum(features, axis=0)
    fltr = np.where(tokenztotalcounts >= 10)[0]
    thefeatures = features.take(fltr, axis=1)
    return thefeatures

##### PART 2
#DONT CHANGE THIS FUNCTION
def part2(X, n_dim):
    #Reduce Dimension
    print("Reducing dimensions ... ")
    X_dr = reduce_dim(X, n=n_dim)
    assert X_dr.shape != X.shape
    assert X_dr.shape[1] == n_dim
    #print("Example sample dim. reduced feature vec: ", X[0])
    print("Dim reduced data shape: ", X_dr.shape)
    return X_dr


def reduce_dim(X,n=10):
    # Before making SVD features invidually scaled to its maximum value
    # transformer = MaxAbsScaler().fit(X)
    # X_scald = transformer.transform(X)
    svd = TruncatedSVD(n_components=n, random_state=42)
    return svd.fit_transform(X)


##### PART 3
#DONT CHANGE THIS FUNCTION EXCEPT WHERE INSTRUCTED
def get_classifier(clf_id):
    if clf_id == 1:
        clf = LinearSVC()
    elif clf_id == 2:
        clf = SGDClassifier()
    else:
        raise KeyError("No clf with id {}".format(clf_id))

    assert is_classifier(clf)
    print("Getting clf {} ...".format(clf.__class__.__name__))
    return clf

#DONT CHANGE THIS FUNCTION
def part3(X, y, clf_id):
    #PART 3
    X_train, X_test, y_train, y_test = shuffle_split(X,y)

    #get the model
    clf = get_classifier(clf_id)

    #printing some stats
    print()
    #print("Train example: ", X_train[0])
    #print("Test example: ", X_test[0])
    #print("Train label example: ",y_train[0])
    #print("Test label example: ",y_test[0])
    print()


    #train model
    print("Training classifier ...")
    train_classifer(clf, X_train, y_train)


    # evalute model
    print("Evaluating classcifier ...")
    evalute_classifier(clf, X_test, y_test)


def shuffle_split(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def train_classifer(clf, X, y):
    assert is_classifier(clf)
    return clf.fit(X, y)


def evalute_classifier(clf, X, y):
    assert is_classifier(clf)
    summary = []
    y_predict = clf.predict(X)
    accuracy = accuracy_score(y, y_predict)
    report = classification_report(y, y_predict, output_dict=True)["weighted avg"]
    summary.append([accuracy, report["precision"], report["recall"], report["f1-score"]])

    # print summary table
    # https://stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
    hdr = ["accuracy", "precision", "recall", "F-measure"]

    myoutput = "{:-^60s}".format("Weighted Avg")+"\n"
    for h in hdr:
        myoutput += "{:^15s}".format(h)
    myoutput += "\n"
    for item in summary[0]:
        myoutput += "{:^15.2f}".format(item)
    myoutput += "\n\n\n"
    
    print(myoutput)
    with open("/home/guszarzmo@GU.GU.SE/LT2212-v20/a2/report1.txt", "a") as f:
        f.writelines(myoutput)


######
#DONT CHANGE THIS FUNCTION
def load_data():
    print("------------Loading Data-----------")
    data = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)
    #print("Example data sample:\n\n", data.data[0])
    #print("Example label id: ", data.target[0])
    #print("Example label name: ", data.target_names[data.target[0]])
    #print("Number of possible labels: ", len(data.target_names))
    return data.data, data.target, data.target_names


#DONT CHANGE THIS FUNCTION
def main(model_id=None, n_dim=False):

    # load data
    samples, labels, label_names = load_data()


    #PART 1
    print("\n------------PART 1-----------")
    X = part1(samples)

    #part 2
    if n_dim:
        print("\n------------PART 2-----------")
        X = part2(X, 
        
        n_dim)

    #part 3
    if model_id:
        print("\n------------PART 3-----------")
        part3(X, labels, model_id)

def clean_text(texts):
    
    # Delete header and signature
    for text in texts:
        textclean = re.sub(regexhdr, '', text)  # delete header
        # remove signature
        nlines = len(textclean.splitlines())
        if nlines >= 17:
            textsplitted = textclean.rsplit("\n",10)
            lasttext = "\n".join(textsplitted[1:10])
            textwosigntr = re.split(regexsignature, lasttext)[0]
            textclean = textsplitted[0] + "\n" + textwosigntr
        yield textclean

def prepare_text(texts):
    # lower case
    for text in texts:
        yield text.lower()

def tokenize_text(texts):
    for text in texts:
        tokenz1 = re.findall(pattern_spesific,text)
        text = re.sub(pattern_spesific, '', text)
        text = re.sub(pattern_specialccsatb, ' ', text)
        tokenz2 = re.findall(alpha,text)
        yield tokenz1+tokenz2

def count_words(texts):
    for text in texts:
        yield Counter(text)

if __name__ == '__main__':
    pass
    parser = argparse.ArgumentParser()
    parser.add_argument("-n_dim",
                        "--number_dim_reduce",
                        default=False,
                        type=int,
                        required=False,
                        help="int for number of dimension you want to reduce the features for")

    parser.add_argument("-m",
                        "--model_id",
                        default=False,
                        type=int,
                        required=False,
                        help="id of the classifier you want to use")

    args = parser.parse_args()
    main(   
            model_id=args.model_id, 
            n_dim=args.number_dim_reduce
            )

