#%%
import a1
import matplotlib.pyplot as plt

class1 = "../../Corpora/reuters-topics/grain"
class2 = "../../Corpora/reuters-topics/crude"
df1 = a1.part1_load(class1, class2, 100)
df2 = a1.part3_tfidf(df1)

# %%
display(df1)


# %%
import numpy as np
X = df1.reset_index(drop=True)
y = df1.index.get_level_values(0).to_series(index=np.arange(X.shape[0]))
#myclass = y.unique()
#y[y==myclass[0]] = 1
#y[y==myclass[1]] = 0
#print(myclass)
display(X)
display(y)

# %%
from sklearn.model_selection import train_test_split
training_data, test_data, training_labels, test_labels = train_test_split(X, y, test_size = 0.20, random_state=0) # Shuffles and splits the data in test and training set
print(test_labels.shape)

# %%
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn import svm
from sklearn import metrics
import numpy as np
X = df1.reset_index(drop=True)
y = df1.index.get_level_values(0).to_series(index=np.arange(X.shape[0]))
CV = StratifiedKFold(n_splits=5, shuffle=True)
clf = svm.SVC(kernel='linear')
scores = cross_val_score(clf, X, y, cv=CV, scoring='accuracy') 
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# %%
