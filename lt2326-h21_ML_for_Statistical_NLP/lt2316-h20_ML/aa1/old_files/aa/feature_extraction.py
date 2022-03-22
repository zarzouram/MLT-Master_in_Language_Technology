import re
from itertools import chain
import pandas as pd

from nltk.stem.lancaster import LancasterStemmer as ls
import spacy
from spacy.tokens import Doc

from .data_loading import DataLoader, pad_post_sequence
from .custom_classes import Vocabulary, DatasetObject
from .ppmi_embedding import ppmi_embedding

import torch
# Feel free to add any new code to this script


class myTokenizer(object):
    # custom spacy tokenization, to produce the same tokens produced by DataLoader class
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        space_id = []
        spaces = []
        # use custom tokenization
        words = re.split(r'(\W)', text.lower())

        # the regular expression produce a trailing "" after each punctuations.
        words = list(filter(lambda w: w != "", words))
        # return whether a word is followed by space, needed by spacy
        space_id = [bool(words[i+1] == " ")
                    for i in range(len(words)-1)] + [False]
        spaces = [space_id[i] for i, x in enumerate(words) if x != " "]

        # remove spaces and "" from tokenization list
        words = " ".join(words).split()
        # assert(len(spaces) == len(words)) # check lengthes
        return Doc(self.vocab, words=words, spaces=spaces)


def assemble_text(tokens, loc):
    text_len = loc[-1][1]
    text = [" "] * text_len

    for token, (i, j) in zip(tokens, loc):
        text[i:j] = token

    return "".join(text[9:-7]).strip()


def get_ling_features(text, special_tokens):

    # initialize linguistics feature vocabulary
    pad_token = special_tokens[0]
    start_sent_token = special_tokens[1]
    end_sent_token = special_tokens[2]

    pos_vocab.add(pad_token)
    dep_vocab.add(pad_token)
    stem_vocab.add(pad_token)
    pos_vocab.add(start_sent_token)
    dep_vocab.add(start_sent_token)
    stem_vocab.add(start_sent_token)
    pos_vocab.add(end_sent_token)
    dep_vocab.add(end_sent_token)
    stem_vocab.add(end_sent_token)

    # extract linguistics features
    sent_pos = []
    sent_dep = []
    sent_stem = []
    sent_head = []
    sent_head_pos = []
    doc = nlp(text)
    for token in doc:
        sent_pos.append(pos_vocab.add(token.pos_))
        sent_dep.append(dep_vocab.add(token.dep_))
        sent_stem.append(stem_vocab.add(ls().stem(str(token.text))))
        sent_head.append(vocab[token.head.text])
        sent_head_pos.append(pos_vocab.add(token.head.pos_))

    # add startseq and endseq tokens
    strt = special_tokens[1]
    end = special_tokens[2]
    sent_pos = [pos_vocab.vocab[strt]] + sent_pos + [pos_vocab.vocab[end]]
    sent_dep = [dep_vocab.vocab[strt]] + sent_dep + [dep_vocab.vocab[end]]
    sent_stem = [stem_vocab.vocab[strt]] + sent_stem + [stem_vocab.vocab[end]]
    sent_head = [vocab[strt]] + sent_head + [vocab[end]]
    sent_head_pos = [pos_vocab.vocab[strt]] + \
        sent_head_pos + [pos_vocab.vocab[end]]

    return [sent_pos, sent_dep, sent_stem, sent_head, sent_head_pos]


def get_features_arr(data_dict: dict, device: torch.device, ppmi=False):

    special_tokens = ["padpad", "startseq", "endseq"]
    pad_token = special_tokens[0]

    seq_tokens_id = data_dict["tokens"]
    seq_tokens_loc = data_dict["loc"]

    # extract linguistics features for each sent
    pos = []
    dep = []
    stem = []
    head = []
    head_pos = []
    for tokens_id, loc in zip(seq_tokens_id, seq_tokens_loc):
        tokens = [vocab_reverse[id_] for id_ in tokens_id]  # decode words
        sentence = assemble_text(tokens, loc)   # reassemble the text
        # extract linguistic features
        ling_X = get_ling_features(sentence, special_tokens)

        pos.append(ling_X[0])
        dep.append(ling_X[1])
        stem.append(ling_X[2])
        head.append(ling_X[3])
        head_pos.append(ling_X[4])

    if ppmi:
        # calculate PPMI then reduce the dimention to D
        vocab_size = vocab.__len__()
        window_size = 4  # 2 on right and 2 on left
        d = 3000
        ppmi, _ = ppmi_embedding(seq_tokens_id, vocab_size, window_size, d, 1)

        return zip(pos, dep, stem, head, head_pos, ppmi)

    else:
        lens = data_dict["length"] +  [max_len]
        pos_padd = pad_post_sequence(pos, lens, pos_vocab.vocab[pad_token])
        dep_padd = pad_post_sequence(dep, lens, dep_vocab.vocab[pad_token])
        stem_padd = pad_post_sequence(stem, lens, stem_vocab.vocab[pad_token])
        head_padd = pad_post_sequence(head, lens, vocab[pad_token])
        head_pos_padd = pad_post_sequence(
            head_pos, lens, pos_vocab.vocab[pad_token])

        # convert to tensor
        pos_padd = torch.tensor(
            pos_padd, dtype=torch.int16, device=device)

        dep_padd = torch.tensor(
            dep_padd, dtype=torch.int16, device=device)

        stem_padd = torch.tensor(
            stem_padd, dtype=torch.int16, device=device)

        head_padd = torch.tensor(
            head_padd, dtype=torch.int16, device=device)

        head_pos_padd = torch.tensor(
            head_pos_padd, dtype=torch.int16, device=device)

    return torch.stack([pos_padd,   dep_padd, 
                        stem_padd,  head_padd, head_pos_padd], 2)


def compare_df(df1, df2):
    arr1 = df1.values[:, 1:]
    arr2 = df2.values
    assert (arr1 == arr2).all()
    print("done")


def extract_features(data: pd.DataFrame, max_sample_length: int,
                     dataset: DataLoader, assignment_1=True):
    # this function should extract features for all samples and
    # return a features for each split. The dimensions for each split
    # should be (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH, FEATURE_DIM)
    # NOTE! Tensors returned should be on GPU
    #
    # NOTE! Feel free to add any additional arguments to this function. If so
    # document these well and make sure you dont forget to add them in run.ipynb

    # global variables
    global pos_vocab, dep_vocab, stem_vocab, vocab_reverse, vocab, nlp, max_len
    # Vocabularies
    pos_vocab = Vocabulary()
    dep_vocab = Vocabulary()
    stem_vocab = Vocabulary()
    vocab = dataset.word_vocab.vocab
    vocab_reverse = dataset.word_vocab.reverse_vocab
    max_len = dataset.max_sample_length

    print("Loading spacy ...")
    # initialize spacy
    nlp = spacy.load("en")
    nlp.tokenizer = myTokenizer(nlp.vocab)
    print("Loading Finished.\n")

    # get sequences
    train_seqs = dataset.train
    val_seqs = dataset.val
    test_seqs = dataset.test

    device = dataset.device
    print("Get features for train dataset ...")
    train_X = get_features_arr(
        data_dict=train_seqs, device=device, ppmi=not assignment_1)

    print("Get features for val dataset ...")
    val_X = get_features_arr(
        data_dict=val_seqs, device=device, ppmi=not assignment_1)

    print("Get features for test dataset ...")
    test_X = get_features_arr(
        data_dict=test_seqs, device=device, ppmi=not assignment_1)

    print("Finish.")

    return train_X, val_X, test_X