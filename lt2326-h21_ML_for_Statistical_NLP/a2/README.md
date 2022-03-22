# Assignment 2: More Chinese-language experiments

## Introduction

The [Demo2.1_OriginalModified_1.ipynb
notebook](a2/Demo2.1_OriginalModified_1.ipynb) is a revised version of the
original notebook "Demo 2.1 - Chinese word segmentation - LSTM.ipynb". I did
the following changes:

  1. removing the additional cells used during the coding session
  2. evaluation part: changes in coding style, keeping everything in one loop, and run evaluation sample by sample.


I created another revised notebook "[Demo2.1_OriginalModified_2.ipynb notebook](a2/Demo2.1_OriginalModified_2.ipynb)" which is a revised version of [Demo2.1_OriginalModified_1.ipynb notebook](a2/Demo2.1_OriginalModified_1.ipynb. The second revised notebook is the basis I used in my solution for Part 1 and Part 2. I did the following changes:

  1. Add Start Of Sentence tokem `<SOS>` and End Of Sentence token `<EOS>`. This change affects `read_chinese_data`, `convert_sentence`, and `index_chars` functions
  2. Add an LSTM encoder layer `self.encoder` above the segmentation layer `self.word_seg`
  3. Use of `CrossEntropyLoss` loss function insread of `NLLLoss`

The performance in the second version notebook is comparable with the original
notebook.

## Part 1 - Sentence generation

Mainly, I separated the embedding layer and added a char generation layer above
the sequence emcoder layer. The model is shown below.

![Model
Architecture](https://github.com/zarzouram/LT2326-H21/blob/main/a1/images/Arch.svg)

Changes are reported and discussed inside the [Demo2.1_Part1_SentsGen notebook](a2/Demo2.1_Part1_SentsGen.ipynb).

## Part 2 - Dual objectives

Changes are reported and discussed inside the [Demo2.1_Part1_SentsGen
notebook](a2/Demo2.1_Part1_SentsGen.ipynb).

## Part 3 - Analysis:

### Word Segmentation Layer

The original objective: [Demo2.1_OriginalModified_2](a2/Demo2.1_OriginalModified_2.ipynb)\
The dual Objectves: [Demo2.1_Part2_DualObjectives](a2/Demo2.1_Part2_DualObjectives.ipynb)

In the original objective notebook, the models take only **30 epochs** to reach
an accuracy of 94.4% and an F1 score of 97%. While in the dual objectives
notebook, the model takes **50 epochs** to get an accuracy of 91.68% and an F1
score of 95%. These results indicate that the dual objectives model takes
slightly longer to reach meaningful results.

The word segmentation layer in the dual objective model takes longer because it
may need to react to the changes in the input weights from the base sequence
encoder. These input weights are affected by the gradients changes coming from
the char generation layer, which is still in the training phase; not
sufficiently converged yet.

### Character Generation Layer

Model perplexity can be correlated to cross-entropy loss. Thus, I will use loss
in the comparison.

The Single objective: [Demo2.1_Part1_SentsGen.ipynb](a2/Demo2.1_Part1_SentsGen.ipynb)\
The dual Objectves: [Demo2.1_Part2_DualObjectives](a2/Demo2.1_Part2_DualObjectives.ipynb)

**For the same number of epochs**(500 epochs), the average loss in the dual
objectives model is less than the single objective model. One can argue that
learning the word boundaries helped enhance the performance of the character
generation.
