import numpy as np
import pandas as pd
import torch
import random
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

class Net(nn.Module):
    def __init__(self, inputsize, hiddenlayer=None, nonlinear=None):
        super(Net, self).__init__()
        self.hiddensize = None
        self.nonlinear = nonlinear
        if hiddenlayer:
            self.fc1 = nn.Linear(inputsize, hiddenlayer)
            self.fc2 = nn.Linear(hiddenlayer, 1)
            self.hiddensize = hiddenlayer
        else:
            self.fc1 = nn.Linear(inputsize, 1)

    def forward(self, x):
        x = self.fc1(x)
        if self.nonlinear == 'Relu':
            x = F.relu(x)
        elif self.nonlinear == "tanh":
            x = F.tanh(x)
        
        if self.hiddensize:
            x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

def batch_data(data, n):
    for i in range(0, len(data), n):
        yield data[i:i+n]

def train(net, trainloader, epochs=3, lr=0.01, momentum=0.3):
    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr, momentum)
    p = int(len(trainloader)/5)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, label = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = net(inputs)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % p == p-1:
                print('[%d, %5d] loss: %.3f' %
                    (epoch+1, i+1, running_loss / p))
                running_loss = 0.0
    
    print('Finished Training')

def test(net, testloader):
    testresults = []
    with torch.no_grad():
        for features in testloader:
            # features, label = data
            outputs = net(features)
            predicted = [1 if output>=0.5 else 0 for output in outputs]
            testresults.extend(predicted)
    return testresults

def evaluate(models, ys, yspredict):
    summaries = []
    for i, (y, ypredict) in enumerate(zip(ys, yspredict)):
        accuracy = accuracy_score(y, ypredict)
        report = classification_report(y, ypredict, output_dict=True)[
            "weighted avg"]
        summaries.append([models[i], accuracy, report["precision"],
                        report["recall"], report["f1-score"]])

    # print summary table
    # https://stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
    hdr = ["model", "accuracy", "precision", "recall", "F-measure"]

    myoutput = "{:-^70s}".format("Weighted Avg")+"\n"
    for i, h in enumerate(hdr):
        if i == 0:
            myoutput += "{:^10s}".format(h)
        else:
            myoutput += "{:^15s}".format(h)
    myoutput += "\n"
    for summary in summaries:
        for item in summary:
            if isinstance(item, str):
                myoutput += "{:^10s}".format(item)
            else:
                myoutput += "{:^15.2f}".format(item)
        myoutput += "\n"
    
    # print(myoutput)
    return myoutput


if __name__ == "__main__":
    dataimported = pd.read_excel("SampleData.xlsx", sheet_name="Train Set", index_col=0).values
    testsetimported = pd.read_excel("SampleData.xlsx", sheet_name="Test Set", index_col=0).values
    
    # train data
    inputs = torch.FloatTensor(dataimported[:, :-1])
    labels = dataimported[:,-1].astype(np.float32)
    labels = np.reshape(labels, (-1,1))
    labels = torch.from_numpy(labels)
    traindata = zip(inputs,labels)

    # test data
    testfeatures = torch.FloatTensor(testsetimported[:, :-1])
    testlabels = testsetimported[:,-1].astype(int).tolist()

    # Intialize, train and test the NN
    net = Net(inputs.shape[1])
    train(net, traindata)
    results = test(net, testfeatures)
    models = ["1","2","3","4"]
    xx = []
    evaluate(models, [testlabels]*4, [results]*4)
    
