import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex
from spacy.lang.char_classes import HYPHENS
from spacy.lang.char_classes import ALPHA
import re

import itertools as it
import pandas as pd

# create a spaCy tokenizer
spacy.load('en')
nlp = spacy.load('en')
nlp.max_length = 1030700
# edit spacy tokenizer
infixes = list(nlp.Defaults.infixes)
infixes.remove(r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS))
infix_regex = spacy.util.compile_infix_regex(infixes)
nlp.tokenizer.infix_finditer = infix_regex.finditer

suffixes = nlp.Defaults.suffixes + \
    (r"(?<=[{a}])(?:{h}+)(?!{a})".format(a=ALPHA, h=HYPHENS),)
suffix_regex = spacy.util.compile_suffix_regex(suffixes)
nlp.tokenizer.suffix_search = suffix_regex.search


# remove html entities from docs and
# set everything to lowercase
# def mypreprocessor(docs):
#     authors = []; authorsmsgs = []
#     for doc in docs:
#         msgslistoflist = list(doc.values())[0]
#         author = list(doc.keys())[0]
#         msgstext = " ".join([list(x) for x in zip(*msgslistoflist)][0])
        
#         authors.append(author)
#         authorsmsgs.append(msgstext)

#     return [authors, iter(authorsmsgs)]

# tokenize the doc and lemmatize its tokens
def mytokenizer(text):

    # Tokenize text producing unigrams and bigrams
    doc = nlp(text, disable=["tagger", "parser"])

    # Merge multi tokens entity to form one token
    with doc.retokenize() as retokenizer:
        for ent in doc.ents:
            retokenizer.merge(doc[ent.start:ent.end])

    unigram = [token.text if token.ent_type_ else token.text.lower()
               for token in doc if not (not token.ent_type_ and token.like_num)]
    # for token in unigram:
    #     print(token)

    token1, token2 = it.tee(unigram)
    next(token2, None)
    bigram = [' '.join(bigram) for bigram in zip(token1, token2)]
    ngrams = unigram + bigram

    return ngrams

