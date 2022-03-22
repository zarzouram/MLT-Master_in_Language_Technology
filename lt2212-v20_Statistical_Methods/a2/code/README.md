# LT2212 V20 Assignment 2

## Part1 - Creating the Feature Table

Text is prepared as follows:

1. Header and signature are detected and then removed from each file by regular expression (`patternhdr` and `patternsignature`).
2. Abbreviations, URLs, times, email domains (@words.com), numbers with percentages and $ character and words that have an underscore, apostrophe, an dash are parsed by regular expression (`pattern_spesific`).
3. Other punctuations and special characters are removed.
4. Words are then tokenized by spaces.
5. Each tokenized word is lowercased

---

## Part 2 - Dimensionality reduction

Singular value decomposition (SVD).\
Note: Principal components analysis (PCA) needs a dense matrix and does not deal with the sparse matrix, which in our case.

---

## Part 4 - Try and discuss

Classifications used are: Linear Support Vector and Stochastic gradient descent. Models report are as follows:

|               |                   | 5%   | 10%  | 25%  | 50%  | 100% |
| ------------- | ----------------- | ---- | ---- | ---- | ---- | ---- |
| **Accuracy**  | **LinearSVC**     | 0.72 | 0.74 | 0.78 | 0.79 | 0.79 |
|               | **SGDClassifier** | 0.65 | 0.68 | 0.70 | 0.71 | 0.74 |
| **Precision** | **LinearSVC**     | 0.72 | 0.75 | 0.78 | 0.79 | 0.79 |
|               | **SGDClassifier** | 0.70 | 0.73 | 0.74 | 0.74 | 0.76 |
| **Recall**    | **LinearSVC**     | 0.72 | 0.74 | 0.78 | 0.79 | 0.79 |
|               | **SGDClassifier** | 0.56 | 0.68 | 0.70 | 0.71 | 0.74 |
| **F-measure** | **LinearSVC**     | 0.72 | 0.74 | 0.78 | 0.79 | 0.79 |
|               | **SGDClassifier** | 0.66 | 0.68 | 0.69 | 0.71 | 0.74 |
|               |

In general, SVM has a better performance than the SGD classifier. The above table shows that ---for the SVM classifier--- the model which has been trained by 50% reduced feature performs as well as the one who has been trained by the unreduced features. For the same classifier, there is also a slight difference in performance between the model that was trained by 25% reduced features and the full features model.

For the SGD classifier, the model performnce increases with the increases of the dimension of features.

## Part 5 - Bonus

The new Sparse Principal Components Analysis. It is noticed fro the table below that the performance of SGD is degraded when it was trained by non-reduced features. This decrease in performance may be due to the separation and shuffling of the data. Normally, I would shuffle and separate the data into k-folds data sets and get the average of the model performance across those data folds. It worth mentioning that the SGD is sensitive to data scale, the feature has not been scaled in this assignment.

|               |                   | 5%     | 10%  | 25%  | 50%  | 100%   |
| ------------- | ----------------- | ------ | ---- | ---- | ---- | ------ |
| **Accuracy**  | **LinearSVC**     | 0.72   | 0.74 | 0.78 | 0.78 | 0.79   |
|               | **SGDClassifier** | 0.66   | 0.69 | 0.67 | 0.69 | `0.64` |
| **Precision** | **LinearSVC**     | 0.72   | 0.75 | 0.78 | 0.79 | 0.79   |
|               | **SGDClassifier** | 0.70   | 0.73 | 0.75 | 0.71 | `0.70` |
| **Recall**    | **LinearSVC**     | 0.72   | 0.74 | 0.78 | 0.78 | 0.79   |
|               | **SGDClassifier** | `0.66` | 0.69 | 0.67 | 0.69 | `0.64` |
| **F-measure** | **LinearSVC**     | 0.72   | 0.74 | 0.78 | 0.79 | 0.79   |
|               | **SGDClassifier** | 0.66   | 0.70 | 0.66 | 0.69 | `0.62` |
|               |