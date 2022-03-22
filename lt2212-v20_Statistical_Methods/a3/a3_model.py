import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
import itertools as it
#import numpy.random
import random
import NN
from sklearn.model_selection import train_test_split
import time
from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# Whatever other imports you need
class MyDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# You can implement classes and helper functions here too.
def split_data(features, samplessize=0):
    docsfeatures = [a[:-1] + b[:-1] + [int(a[-1] == b[-1])]
                    for a, b in it.combinations(features, 2)]

    docsfeatures = np.array(docsfeatures)
    
    Xtrain, Xtest, ytrain, ytest = train_test_split(
            docsfeatures[:, :-1], docsfeatures[:, -1], 
            test_size = 0.2, 
            stratify = docsfeatures[:, -1], 
            random_state = 42, 
            shuffle = True
            )

    # balance classes .. sameauthors=diffauthors
    # trainset = np.hstack((Xtrain, np.array([ytrain]).T))
    # testset = np.hstack((Xtest, np.array([ytest]).T))
    # sameauthors = trainset[trainset[:,-1] == 1] #filter sameauthors class
    # diffauthors = trainset[trainset[:,-1] == 0] #filter diffauthors class
    # s1, s2 = sameauthors.shape[0], diffauthors.shape[0]

    # samples = []; idx1= set(); idx2 = set()
    # size = samplessize if samplessize else min(s1 , s2)*2
    # for _ in range(size):
    #     if random.random() > 0.5:
    #         while True:
    #             idx = np.random.choice(s1, size=1, replace=False)
    #             idx = idx.tolist()[0]
    #             if idx not in idx1:
    #                 samples.append(sameauthors[idx].tolist())
    #                 idx1.add(idx)
    #                 break
    #     else:
    #         while True:
    #             idx = np.random.choice(s2, size=1, replace=False)
    #             idx = idx.tolist()[0]
    #             if idx not in idx2:
    #                 samples.append(diffauthors[idx].tolist())
    #                 idx2.add(idx)
    #                 break

    # Preparing train and test dataset to be suitable for network use.
    # samples = np.array(samples) 
    # trainsamples = torch.FloatTensor(samples[:, :-1])
    # trainlabels = samples[:,-1].astype(np.float32)
    # trainlabels = np.reshape(trainlabels, (-1,1))
    # trainlabels = torch.from_numpy(trainlabels)
    # traindata = list(zip(trainsamples, trainlabels))

    # testfeatures = torch.FloatTensor(testset[:, :-1])
    # testlabels = testset[:,-1].astype(int).tolist()
    #export to excel
    # trainsetdf = [pd.DataFrame(samples)]
    # testsetdf = [pd.DataFrame(testset)]
    # with pd.ExcelWriter("SampleData.xlsx") as writer:
    #     for traindf, testdf in zip(trainsetdf, testsetdf):
    #         traindf.to_excel(writer, sheet_name="Train Set")
    #         testdf.to_excel(writer, sheet_name="Test Set")
    #     writer.save()

    # export to csv
    # samplesdf = pd.DataFrame(samples)
    # testdf = pd.DataFrame(testset)
    # samplesdf.to_csv("train_s.csv", index=False)
    # testdf.to_csv("test_s.csv", index=False)
    # return (traindata, testfeatures, testlabels)
    return [ Xtrain, Xtest, ytrain, ytest ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefile", type=str, help="The file containing the table of instances and features.")
    parser.add_argument("--test", "-T", dest="samplesize", type=int, default="0", help="The size of samples used to train the network. 0 to use all data.")
    parser.add_argument("--hidden", "-D", dest="hiddenlayer", type=int, default="0", help="The size of hidden layer.")
    parser.add_argument("--nonlinearity", "-L", dest="nlinearity", type=str, default=None, help="Network Nonlinearity type.")
    # Add options here for part 3 -- hidden layer and nonlinearity,
    # and any other options you may think you want/need.  Document
    # everything.
    
    args = parser.parse_args()
    filepath = args.featurefile
    samplesize = args.samplesize
    hiddenlsize = args.hiddenlayer
    nonlinearity = args.nlinearity

    print("Reading {}...".format(filepath))
    # X = pd.read_excel(args.featurefile, sheet_name="Train Set", index_col=0).values
    # y = pd.read_excel(args.featurefile, sheet_name="Test Set", index_col=0).values
    dataimported = pd.read_csv(filepath, index_col=0)
    X = dataimported.loc[dataimported.Label == "train"]
    y = dataimported.loc[dataimported.Label == "test"]
    X = X.drop("Label", axis=1).values
    y = y.drop("Label", axis=1).values
    print("Finished Reading")
    
    # st1 = time.time()
    print()
    print("Sampling data ...")
    features = np.vstack((X, y))
    # sampling data
    Xtrain, Xtest, ytrain, ytest = split_data(features.tolist(), samplesize)

    classes = [0, 1]
    samplesclasscount = np.array([ytrain[ytrain==t].shape[0] for t in classes])
    weights = 1. / samplesclasscount
    samplesweight = np.array([weights[int(t)] for t in ytrain], dtype=np.float64)
    
    samplesweight = torch.from_numpy(samplesweight)
    Xtrain = torch.FloatTensor(Xtrain); Xtest = torch.FloatTensor(Xtest)
    ytrain = ytrain.astype(np.float32); ytest = ytest.astype(np.int)
    ytrain = np.reshape(ytrain, (-1,1))
    ytrain = torch.from_numpy(ytrain); ytest = ytest.tolist()
    
    samplesnum = samplesize if samplesize else samplesweight.shape[0]
    sampler = WeightedRandomSampler(
        samplesweight, samplesnum, replacement=False)
    
    
    print("Finished Sampling")
    print()
    results = []
    # Intialize, train and test the NN
    net = NN.Net(Xtest.shape[1], hiddenlsize, nonlinearity)
    print("Network Initialized:\n\t", net)

    print("Training network ...")

    traindata = TensorDataset(Xtrain, ytrain)
    traindataloader = MyDataset(traindata)
    batchsize = 100
    train_loader = DataLoader(traindata, batch_size=batchsize,
                            num_workers=6, sampler = sampler, shuffle=False)
    NN.train(net, train_loader)

    print()
    print("Evaluation\n")
    print()
    print("Network:\n\t", net)

    test_loader = DataLoader(Xtest, batch_size=batchsize,
                            num_workers=6, shuffle=False)
    ypredict = NN.test(net, test_loader)
    # et1 = time.time()
    model = str(nonlinearity) + "-" + str(hiddenlsize)
    result = NN.evaluate([model], [ytest], [ypredict])

    print(result)

    # with open("Output.txt", "w") as fo:
    #     fo.write(result)