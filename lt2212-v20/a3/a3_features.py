import os
import glob
import sys
import argparse
import numpy as np
import pandas as pd
from read_emails import get_data
from sklearn.feature_extraction.text import CountVectorizer
from prepare_text import mytokenizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD, PCA

# Whatever other imports you need

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directories into table.")
    parser.add_argument("inputdir", type=str, help="The root of the author directories.")
    parser.add_argument("outputfile", type=str, help="The name of the output file containing the table of instances.")
    parser.add_argument("dims", type=int, help="The output feature dimensions.")
    parser.add_argument("--test", "-T", dest="testsize", type=int, default="20", help="The percentage (integer) of instances to label as test.")

    args = parser.parse_args()

    print("Reading {}...".format(args.inputdir))
    # Do what you need to read the documents here.
    maindir = args.inputdir
    docspath = glob.iglob("{}/*/*".format(maindir))
    authors = []; msgs = []
    for filepath in docspath:
        msg, author = get_data(filepath)
        authors.append(author)
        msgs.append(msg)

    print("Constructing table with {} feature dimensions and {}% test instances...".format(args.dims, args.testsize))
    # Build the table here.
    vectorizer = CountVectorizer(analyzer=mytokenizer, min_df=10)
    X = vectorizer.fit_transform(msgs)

    # Reduce feature matrix
    svd = TruncatedSVD(n_components=args.dims)
    Xreduced = svd.fit_transform(X)
    authors = np.array(authors)

    # Split data
    nsplit = args.testsize/100
    # trainsets = []; testsets = []
    colsname = ["PC{0!s}".format(pc) for pc in range(1, args.dims+1)]+["Target"]+["Label"]
    
    Xtrain, Xtest, ytrain, ytest = train_test_split(
            Xreduced, authors, 
            test_size = nsplit, 
            stratify = authors, 
            random_state = 42, 
            shuffle = True
            )

    datalabelstrain = np.array([["train"]]*Xtrain.shape[0])
    datalabelstest = np.array([["test"]]*Xtest.shape[0])
    trainset = np.concatenate(
        (Xtrain, np.array([ytrain]).T, datalabelstrain), axis=1)
    testset = np.concatenate(
        (Xtest, np.array([ytest]).T, datalabelstest), axis=1)
    dataset = np.vstack((trainset, testset))
    dataset = pd.DataFrame(dataset, columns=colsname)
    # trainsets.append(pd.DataFrame(trainset, columns=colsname))
    # testsets.append(pd.DataFrame(testset, columns=colsname))

    print("Writing to {}...".format(args.outputfile))

    # Write the table out here.
    # with pd.ExcelWriter(args.outputfile+".xlsx") as writer:
    #     for traindf, testdf in zip(trainsets, testsets):
    #         traindf.to_excel(writer, sheet_name="Train Set")
    #         testdf.to_excel(writer, sheet_name="Test Set")
    #     writer.save()
    dataset.to_csv(args.outputfile)
    print("Done!")
    
    
