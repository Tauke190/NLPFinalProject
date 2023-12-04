import spacy;
import re
from sacremoses import MosesTokenizer, MosesDetokenizer
from nltk.corpus import brown
import numpy as np
from spacy.util import compile_infix_regex
from spacy.tokenizer import Tokenizer


nlp = spacy.load("en_core_web_sm", disable=["parser"])
boundary = re.compile('^[0-9]$')


doc_names = brown.fileids()
tweets = []           # Prediction --- List(doc)
tweets_segmented = [] # Answer key--- List[List(doc)]

def sent_indices_from_list(sents, space_included=False):
    """
    Convert a list of sentences into a list of indiceis indicating the setence spans.
    For example:
    Non-tokenised text: "A. BB. C."
    sents: ["A.", "BB.", "C."] -> [3, 7]
    """
    indices = []
    offset = 0
    for sentence in sents[:-1]:
        offset += len(sentence)
        if not space_included:
            offset += 1
        indices += [offset]
    return indices

def evaluate_indices(true, pred):
    """
    Calculate the Precision, Recall and F1-score. Input is a list of indices of the sentence spans.
    """
    true, pred = set(true), set(pred)
    TP = len(pred.intersection(true))
    FP = len(pred - true)
    FN = len(true - pred)
    return TP, FP, FN

def score(TP, FP, FN):
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2*(precision*recall)/(precision+recall) if precision+recall!=0 else 0
    return precision, recall, f1

with open('tweets.txt') as file:
    data = file.read()
    tweets = data.split('\n')

with open('tweets_segmented.txt') as file:
    data = file.read()
    paragraphs = data.split('\n\n')
    for paragraph in paragraphs:
        tweets_segmented.append(paragraph.split('\n'))
    
natural_tweets = [' '.join(doc) for doc in tweets_segmented]


# For ellipses ....
# infixes = [x for x in nlp.Defaults.infixes if x != r'\.\.\.+']
# infix_regex = compile_infix_regex(infixes)
# nlp.tokenizer = Tokenizer(nlp.vocab, infix_finditer=infix_regex.finditer)

# Determine sentence indicies. List[List[int]] ---> Answer_key
tweets_sent_indices = [sent_indices_from_list([sent for sent in doc]) for doc in tweets_segmented]

# Store total length of each document
total_len = [len(' '.join(sent for sent in doc)) for doc in tweets_segmented]

# Boundry as : ! , ; , ?
@spacy.Language.component("custom_boundry")
def set_period_boundaries(doc):
    for token in doc[:-1]:
        # Implement your rules here
        # Example: If token is a period, set next token as a new sentence start
        if token.text == ".":
            doc[token.i + 1].is_sent_start = True
        if token.text == ":":
            doc[token.i + 1].is_sent_start = True
        if token.text == '...':
             doc[token.i + 1].is_sent_start = True
        elif token.text == "?":
            doc[token.i + 1].is_sent_start = True 
        elif token.text == ";":
            doc[token.i + 1].is_sent_start = True
        elif token.text == "!":
            doc[token.i + 1].is_sent_start = True
        elif token.text == "#":
            doc[token.i].is_sent_start = True
        else:
            pass
    return doc

@spacy.Language.component("multiple_spaces")
def multiple_spaces(doc):
    for token in doc[:-1]:
        if token.text.isspace() and doc[token.i + 1].is_alpha:
            doc[token.i + 1].is_sent_start = True
        else:
            # Explicitly set sentence start to False for other tokens
            doc[token.i + 1].is_sent_start = False
    return doc

# nlp.add_pipe("multiple_spaces",before='ner')
nlp.add_pipe("sentencizer",before='ner')
nlp.add_pipe("custom_boundry",before='ner')


# #--------------------------------------------------------------------------------------------------->>
total_tweets_to_test = 20
# Prediction for the tweets
spacy_docs = [nlp(text) for text in tweets[:total_tweets_to_test]]
# [List[List(str)]]
spacy_sents_str = [[sents.text_with_ws for sents in doc.sents] for doc in spacy_docs]
spacy_sent_indicies = [sent_indices_from_list(doc, space_included=True) for doc in spacy_sents_str]


# print(spacy_sents_str[0])
# print(tweets_segmented[0])

# print(spacy_sents_str[1])
# print(tweets_segmented[1])

print(spacy_sents_str[12])
print(tweets_segmented[12])


# Answer key from the brown corpos
spacy_metrics = np.array([evaluate_indices(tweets_sent_indices[i], spacy_sent_indicies[i])
                   for i in range(total_tweets_to_test)])

spacy_metrics_avg = score(*spacy_metrics.sum(axis=0))
print("Precision: %.3f" % spacy_metrics_avg[0])
print("Recall: %.3f" % spacy_metrics_avg[1])
print("F1-score: %.3f" % spacy_metrics_avg[2])