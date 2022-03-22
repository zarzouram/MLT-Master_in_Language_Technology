# Assignment 1: Chinese character "detection"

- [1. Summary](#1-summary)
- [2. Requirements](#2-requirements)
- [3. Part 1: data preparation](#3-part-1-data-preparation)
  - [3.1. Description](#31-description)
  - [3.2. Running the code](#32-running-the-code)
- [4. Part 2: the models](#4-part-2-the-models)
  - [4.1. General](#41-general)
  - [4.2. Model 1](#42-model-1)
  - [4.3. Model 2](#43-model-2)
  - [4.4. Training and evaluation](#44-training-and-evaluation)
    - [4.4.1. Hyperparameters](#441-hyperparameters)
    - [4.4.2. Description](#442-description)
    - [4.4.3. Running the code](#443-running-the-code)
- [5. Part 3: testing and evaluation](#5-part-3-testing-and-evaluation)
  - [5.1. Validation](#51-validation)
  - [5.2. Tetsing](#52-tetsing)
  - [5.3. Results and discussion](#53-results-and-discussion)
- [6. Bonus A: detecting a specific character](#6-bonus-a-detecting-a-specific-character)
- [7. Suggestion to improve the performance:](#7-suggestion-to-improve-the-performance)

## 1. Summary

I chose not to reduce the image size, this choice introduced constraints to the model
design, hyperparameters setting, and validation running time. It seems that the models are
not deep enough to extract the required latent information from images. That is
why the models have such poor performance.

Initially, I trained each model for about 100 epochs to examine their
behavior. I found that the learning rate needs to be changed during
the training period. So, a custom scheduler has been implemented to change the
learning rate if the validation metric did not change for two validation
iterations. You can find more on that under section [4.4.1.3. **Learning
rate**](#4413-learning-rate).

*IMPORTANT NOTE*: Before running the code you need to run `vidom` package. Just
write `visdom` in the terminal.

Ysou will find the best models under `/scratch/<USERNAME>/lt2326_labs/lab1/checkpoints`

## 2. Requirements

Required packages are at `LT2326-H21/a1/requirements.txt`.

  * h5py v3.4.0 used to save processed dataset
  * opencv v4.5.3 used to load images
  * scikit-learn v 1.0 used to calculation different metrics
  * pyTorch used for constructing/training NN platform
  * visdom to plot training and validation metrics

## 3. Part 1: data preparation

You can find all codes that are related to dataset preparation under
`/home/guszarzmo@GU.GU.SE/LT2326/Assignments/LT2326-H21/a1/code/dataset`. The
images are loaded, labeled, and then split as follows:

   1. 70% for models training
   2. 15% for models validation
   3. 15% for models testing

Also, the mean and standard deviation of the images in the training set are
calculated to be used in image normalization.

### 3.1. Description

The code `dataset_prep.py` load images and annotations files and produce a
label matrix for each image as following:

   1. Pixels that are `is_chinese` have label `1`
   2. Pixels that belong to `ignore` boundary box or not-`is_chineses` have
      label `2`
   3. Pixels that are annotated to be a certain character have label `3` (Bonus-B)
   4. all other pixels have label `0`

After processing, for each split, images and labels in `np.array` are writtrn to an h5 file.

### 3.2. Running the code

```bash
python dataset_prep.py [OPTIONS]
```
The `dataset_prep.py` takes the following optinal arguments:

  1. `--dataset_dir`     `-d` \
      Parent directory of the dataset.\
      Default value: `/scratch/lt2326-h21/a1`

  2. `--image_dir`       `-i`\
      Relative directory that contain the images.\
      Default value: `images`

  3. `--train_ann_path`  `-ta`\
      Relative pathe that contain the train annotation jsonl file. \
      Default value: `train.jsonl`

  4. `--val_ann_path`    `-va`\
      Relative path that contain the validation annotation jsonl file. \
      Default value: `val.jsonl`

  5. `--output_dir`      `-o`\
      Output directory. \
      Default value: `/scratch/<USERNAME>/lt2326_labs/lab1/data`

To run the code with default options, use `python dataset_prep.py`

## 4. Part 2: the models

### 4.1. General

In my implementation, I used convolutional and transposed convolutional layers.
Because I used the images in their original size, the pooling, unpooling
layers are depreciated.

### 4.2. Model 1

The first model consists of three convolutional layers that encode the image
into (256, 64, 64) latent space. After that, a series of three transposed convolutional
layers decode the encoded information back to (n, 2048, 2048), where n = 4,
because we have four classes, as shown in section [3.1. Description](#31-description). The model can be found at `LT2326-H21/a1/code/models/simple_model_1.py`

The below figure shows the model configuration.

![Model 1](https://github.com/zarzouram/LT2326-H21/blob/main/a1/images/model_1.png)

### 4.3. Model 2

In this model, I borrowed the idea of "skip connections" from FCN. The idea is
during the encoding process some information may be lost. So, adding
information from encoding to upsampled space may help to enhance the
performance. The model can be found at
`LT2326-H21/a1/code/models/simple_model_1.py`

![Model 2](https://github.com/zarzouram/LT2326-H21/blob/main/a1/images/model_2.png)

### 4.4. Training and evaluation

#### 4.4.1. Hyperparameters

##### 4.4.1.1. **Model layers configuration**:

  Convolutional layers configuratios (for both models)
  1. convolution layers (conv): all layers have 8x8-kernel size, 4-stride, padding-2
  2. transposed convolution layers (tconv): same as convolution layers

  Model layers configuration (for both models)
  1. conv-1: H/4 , W/4 , 32
  2. RELU
  3. conv-2: H/16, W/16, 128
  4. RELU
  5. conv-2: H/64, W/64, 256
  6. RELU
  7. tconv-1: H/16, W/16, 128
  8. RELU
  9. tconv-2: H/4 , W/4 , 32
  10. RELU
  11. tconv-2: H , W , 4
  12. RELU

##### 4.4.1.2. **Batch size**:

16, the maximum batch size I can use without cuda out of memory error.

**Note:** due to constraints that we discussed above in section [1.
Summary](#1-summary), It is hard to change the above parameters.

##### 4.4.1.3. **Learning rate**:

initially, a fixed value of 1e-5 is used. However, the validation metric does
not change, As shown in the figures below. Notice that the training and
validation loss is still changing.

<img
src="https://github.com/zarzouram/LT2326-H21/blob/main/a1/images/plot_valmetric_1e-5.png"
width=75% height=75% alt="Val metric does not change">

<img
src="https://github.com/zarzouram/LT2326-H21/blob/main/a1/images/plot_loss_1e-5.png"
width=75% height=75% alt="Losses do not change">

So, I have used a scheduler that changes the learning rate by adding a fixed value
of 7.5e-6 to the initial rate of 1e-5 if the validation metric does not change
for two epochs. See the figures below. You can find the scheduler at `LT2326-H21/a1/code/utils/scheduler.py`


<img
src="https://github.com/zarzouram/LT2326-H21/blob/main/a1/images/plot_valmetric_1e-5_plat.png"
width=75% height=75% alt="Val metric changes">

<img
src="https://github.com/zarzouram/LT2326-H21/blob/main/a1/images/plot_loss_1e-5_plat.png"
width=75% height=75% alt="Losses change">

#### 4.4.2. Description

I trained both model_1 and model_2 for 350 epochs. I used cross-entropy to
calculate the loss between the target labels and the model output.

Every two epochs, the model is validated using the validation dataset. A
softmax classiﬁer is used to predict the 4-class distribution. I chose to use
`f1-score` as a validation metric which was calculated using scikit-learn
package. The best model that have the higher `f1-score` is saved.

As shown in the figures below, model_2 converges faster than model_1 as the
training loss of model_2 is lower than model_1. However, this is not the case
for the validation metric; see section [5. Part 3: testing and
evaluation](#5-part-3-testing-and-evaluation). Also, from the plotted
validation loss, it is noted that model_2 tends to overfit more than model_1.

<img
src="https://github.com/zarzouram/LT2326-H21/blob/main/a1/images/models_training.png"
width=75% height=75% alt="Modell training loss plot">

#### 4.4.3. Running the code

Before running the code you need to run `vidom` package. Just
write `visdom` in the terminal.

In another terminal use

```bash
python run_train.py [OPTIONS]
```

The `run_train.py` takes the following optinal arguments:

  1. `--dataset_dir`                                            \
      Directory contains the processed dataset (h5 files).     \
      Default value: `/scratch/<USERNAME>/lt2326_labs/lab1/data`

  2. `--checkpoint_dir`                                         \
      Directory to save models duriing training.                \
      Default value: `/scratch/<USERNAME>/lt2326_labs/lab1/checkpoints`

  3. `--model`                                                  \
      Model to train. 1 or 2.                                   \
      Default value: `1`

  4. `--load_model`                                             \
      Checkpoint filename to load. Used to resume taining.      \
      Default value: ``

  5. `--plot_env_name`                                           \
      Visdom env. name to plot the training and validation loss. \
      Default value: `plot`

  6. `--gpu`
     GPU device to be used. To automatically select the id based on the
     available memory use -1
     Default value: `-1`

To run the code with the default options, use `run_train.py --model 1` or `run_train.py --model 2`

## 5. Part 3: testing and evaluation

### 5.1. Validation

Every two epochs, the model is evaluated against the validation dataset. 'F1
score` is used during evaluation. The figure below shows that model_2 overfit
before model_2. Also, model_1 converges faster than model_2.

<img
src="https://github.com/zarzouram/LT2326-H21/blob/main/a1/images/models_testing.png"
width=75% height=75% alt="Val metric plot">

### 5.2. Tetsing

Testing experiments are in Jupiter notebooks `a1/code/experiment_model_1.ipynb` and
`a1/code/experiment_model_2.ipynb`. The best model is loaded from
the checkpoint folder and tested against the testing dataset.

### 5.3. Results and discussion

As the results from the notebooks would show, `model_2` performs better than `model_1` in terms of
f1_score, precision, and recall. It seems that skip connections enhance the
model performance. However, during the validation loop, model_2 did not perform
better than model_1. For example, model_2 converges slower and overfits faster
than model_1. This relatively lousy performance in the validation loop may be because
the data from the encoder stage add noise to the model output.

## 6. Bonus A: detecting a specific character

From the beginning, I have added a classifier to detect whether a particular Chinese character
exists. My approach turns the problem into multiclass classification. In
`a1/code/dataset/dataset_prep.py`, you will find a function called
`slect_char` responsible for selecting the most common character in the
dataset.

The notebooks show that `model_2` is capable to detect the specific chinease
character while `model_1` is not. It seems that skip connections enhance the
model performance. Also, The images superimposing supports this claim.

## 7. Suggestion to improve the performance:

1. Use pre-trained CNN model as encoder and freeze it, then fine-tune the
   pre-trained model when the whole model converges to the solution.
2. Make the model deeper. Either use multi-GPU during training or reduce the
   image from beginning
3. Use multi-task learning (MTL) by separating `is_char` and `is_chinese`
   decoder/classifier and sharing the pre-trained encoder.
