# LT2212 V20 Assignment 3

## Part 1 - creating the feature table

### Running the code

`python a3_features.py <inputdir> <outputpath> <dims> -T <testsize>`

1. `inputdir` is a full directory of enon emails like: `/home/guszarzmo@GU.GU.SE/Corpora/enron-emails/enron_sample`
2. `outputpath` is the output file path containing the features table like `summary.csv`
3. `dims` is the output feature dimensions in `int`
4. `testsize` [optional] is the percentage of instances to label as a test in `integer`.

**example**: `python a3_features.py /home/guszarzmo@GU.GU.SE/Corpora/enron-emails/enron_sample summary.csv 120 -T 20`

### Text preprocessing

Text is prepared as follows:

1. Email files are parsed by the `email` package. The header is then removed, keeping only the email body message. Parsing is also done for the original emails that are attached to the main email as a reply or forwarded message.
2. Signature is removed from each body message by the regular expression.
3. Text is tokenized by `Spacy` package. The `infixes` and `suffixes` in `Spacy` are modified to tokenize words that have internal hyphens as one token.
4. Other punctuations and special characters are removed.
5. Each tokenized word is lowercased if it is not detected as an entity by `Spacy`.
6. All numbers are removed if they are not entity e.g., time or money.

### Features

1. Features are constructed by counting unigrams and bigrams using `sklearn.feature_extraction.text.CountVectorizer`.
2. Features dimension are then reduced according to the parsed argument `args.dims` using `sklearn.decomposition.TruncatedSVD`. This method deals with the sparse matrix.
3. Features matrix is then split into training and test set according to the parsed argument `args.testsize` using `sklearn.model_selection.train_test_split`. The function ensures that the data is split so that the class ratios in the original data are preserved in the split dataset.

## Part 2 and 3 - design and train models

### Running the code

`python a3_model.py <featurefile> -T <samplesize> -D <hiddenlayer> - L <nlinearity>`

1. `featurefile` is the file path that contains the model features
2. `samplesize` [optinal] is the size of samples that used to train the model.
3. `hiddenlayer` [optional] is the size of the network hidden layer in `int`. Zero is the default meaning that there is no hidden layer.
4. `nlinearity` [optional] is network nonlinearity type, either `Relu` or `tanh`. `None` is the default value meaning that there is no nonlinearity in the network.

**examples**\
`python a3_model.py summary.csv -T 10000 -D 200 -L Relu`\
`python a3_model.py summary.csv -T 10000`\
`python a3_model.py`,*the code runs forever*

### Features and features sampling

The class definition in this part has been changed from what had been defined in part one. In part one, each author constructs a class. Here, the classes are either 0 if two documents have different authors or one if they have the same authors. The features are also redefined here; the new features are now constructed from the features of two documents.

A new features matrix has been constructed from the imported file, by combining each row with all successive rows to construct a new feature. If the combined rows have the same author, they are added to class one; otherwise, the newly constructed feature is added to class zero. After that, the data is split again as discussed in part 1 to preserve the class ratio, founded in the original dataset, in the train set and test set.

To sample data `torch.utils.data.WeightedRandomSampler` which samples data according to a given probability (weight).

### Network training

`torch.utils.data, DataLoader` is used to load data and feed it in batches. The following network was built, trained, and tested as follows:

1. Linear network with no hidden layer
2. Network with one hidden layer of size 200 and Relu
3. Network with one hidden layer of size 100 and Relu
4. Network with one hidden layer of size 200 and tanh
5. Network with one hidden layer of size 100 and tanh

Number of epochs is three and data sample size is 20000. The results are as follows:

|   model  | accuracy | precision | recall | F-measure |
|:--------:|:--------:|:---------:|:------:|:---------:|
| Linear   |   0.67   |    0.74   |  0.67  |    0.70   |
| Relu-200 |   0.55   |    0.74   |  0.55  |    0.61   |
| Relu-100 |   0.62   |    0.74   |  0.62  |    0.66   |
| tanh-200 |   0.50   |    0.76   |  0.50  |    0.56   |
| tanh-100 |   0.55   |    0.76   |  0.55  |    0.61   |

 The results show that the linear network with no hidden network has the best accuracy. It should be considered that the number of epochs is low and may not be enough for other network configurations to provide a better model. Higher accuracy could be a sign of overfitting for the Linear model. More sensible results could be obtained if the training of networks is tuned.
