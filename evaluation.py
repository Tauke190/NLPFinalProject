import re
import nltk
import numpy as np
from nltk.corpus import brown
from sacremoses import MosesTokenizer, MosesDetokenizer

nltk.download('brown', quiet=True)

doc_names = brown.fileids()

# Reconstruct natural text
detokenizer = MosesDetokenizer()

# Gives a list of documents which contains a list of setneces. List[List[str]]
brown_natural_docs_sents = [
    [
        detokenizer.detokenize(
        ' '.join(sent)\
            .replace('``', '"')\
            .replace("''", '"')\
            .replace('`', "'")\
            .split()
        , return_str=True)
        for sent in brown.sents(doc)
    ]
    for doc in doc_names
]

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


# true_sentences = sent_indices_from_list(brown_natural_docs_sents[0])
# predicted_sentences = sent_indices_from_list(brown_natural_docs_sents[0])
# TP,FP,FN = evaluate_indices(true_sentences,predicted_sentences)
# precision,recall,f1 = score(TP,FP,FN)
# print(precision,recall,f1)


# Reproduce documents by joining senteces together. List[str]
brown_natural_docs = [' '.join(doc) for doc in brown_natural_docs_sents]

# Determine sentence indicies. List[List[int]]
brown_sent_indicies = [sent_indices_from_list([sent for sent in doc]) for doc in brown_natural_docs_sents]

# Store total length of each document
total_len = [len(' '.join(sent for sent in doc)) for doc in brown_natural_docs_sents]



def split_on_punct(doc: str):
    """ Split document by sentences using punctuation ".", "!", "?". """
    punct_set = {'.', '!', '?'}
    
    start = 0
    seen_period = False
    
    for i, token in enumerate(doc):        
        is_punct = token in punct_set
        if seen_period and not is_punct:
            if re.match('\s', token):
                yield doc[start : i+1]
                start = i+1
            else:
                yield doc[start : i]
                start = i
            seen_period = False
        elif is_punct:
            seen_period = True
    if start < len(doc):
        yield doc[start : len(doc)]
# Split documents


# [List[List(str)]]
punct_sents_str = [list(split_on_punct(doc)) for doc in brown_natural_docs]


print(punct_sents_str[0])

