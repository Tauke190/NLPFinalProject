import spacy;
import re
from sacremoses import MosesTokenizer, MosesDetokenizer
from nltk.corpus import brown
import numpy as np

nlp = spacy.load("en_core_web_sm")
boundary = re.compile('^[0-9]$')
nlp.disable_pipes('parser')

predicted_sentences_in_docs = []
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



brown_natural_docs = [' '.join(doc) for doc in brown_natural_docs_sents]

# Determine sentence indicies. List[List[int]] ---> Answer_key
brown_sent_indicies = [sent_indices_from_list([sent for sent in doc]) for doc in brown_natural_docs_sents]

# Store total length of each document
total_len = [len(' '.join(sent for sent in doc)) for doc in brown_natural_docs_sents]


# Boundry as : ! , ; , ?
@spacy.Language.component("custom_boundry")
def set_period_boundaries(doc):
    for token in doc[:-1]:
        # Implement your rules here
        # Example: If token is a period, set next token as a new sentence start
        if token.text == ".":
            doc[token.i + 1].is_sent_start = True
        elif token.text == "?":
            doc[token.i + 1].is_sent_start = True 
        elif token.text == ";":
            doc[token.i + 1].is_sent_start = True
        elif token.text == "!":
            doc[token.i + 1].is_sent_start = True    
        else:
            # Other conditions based on your custom rules
            pass
    return doc

# Customer boundry rule , Numbered list like this 1. Hello World = Default splits into "1." && "Hello World"
@spacy.Language.component("numbered_list_with_space")
def custom_seg(doc):
    prev = doc[0].text
    length = len(doc)
    for index, token in enumerate(doc):
        if (token.text == '.' and boundary.match(prev) and index!=(length - 1)):
            doc[index+1].sent_start = False
        prev = token.text
    return doc

@spacy.Language.component("is_abbreviation")
def is_abbreviation(doc):
    for token in doc[:-1]:
        # Default assumption: token is not the start of a sentence
        token.is_sent_start = False

        # Check for potential sentence start
        if token.text == '.':
            # Check for one-letter initial (e.g., "A.")
            if token.i > 0 and len(doc[token.i - 1].text) == 1 and doc[token.i - 1].is_alpha:
                doc[token.i + 1].is_sent_start = False
            # Check for two-letter initial (e.g., "J. K.")
            elif token.i > 2 and doc[token.i - 1].text == '.' and len(doc[token.i - 2].text) == 1 and doc[token.i - 2].is_alpha:
                doc[token.i + 1].is_sent_start = False
            else:
                # Period that is not after an initial
                doc[token.i + 1].is_sent_start = True

    # Explicitly set sentence start for the first token
    doc[0].is_sent_start = True
    return doc

@spacy.Language.component("ignore_num_as_sentence_boundry")
def ignore_num_as_sentence_boundry(doc):
    for token in doc[:-1]:
        # Default assumption: token is not the start of a sentence
        token.is_sent_start = False

        if token.text == '.':
            # Check if the previous token is numeric (part of a number like $100.00)
            if token.i > 0 and (doc[token.i - 1].like_num or doc[token.i - 1].text.startswith('$')):
                # Check if the next token starts with a capital letter
                if token.i + 1 < len(doc) and doc[token.i + 1].text.istitle():
                    doc[token.i + 1].is_sent_start = True
                else:
                    doc[token.i + 1].is_sent_start = False
            else:
                # Period that is not part of a number
                doc[token.i + 1].is_sent_start = True

    # Explicitly set sentence start for the first token
    doc[0].is_sent_start = True
    return doc

@spacy.Language.component("is_inside_paranthesis")
def is_inside_paranthesis(doc):
    open_quote_indices = [tok.i for tok in doc if tok.text == "'"]
    close_quote_indices = [tok.i for tok in doc if tok.text == "'" and tok.i not in open_quote_indices]

    for token in doc[:-1]:
        # Default assumption: token is not the start of a sentence
        token.is_sent_start = False

        if token.text == '.':
            # Check if the period is within any pair of single quotations
            if any(open_quote_index < token.i < close_quote_index for open_quote_index, close_quote_index in zip(open_quote_indices, close_quote_indices)):
                doc[token.i + 1].is_sent_start = False
            else:
                # Period that is not inside single quotations
                doc[token.i + 1].is_sent_start = True

    # Explicitly set sentence start for the first token
    doc[0].is_sent_start = True
    return doc




@spacy.Language.component("set_abbreviation_boundry")
def set_abbreviation_boundry(doc):
    abbreviations = {"E.U."}  # Add more abbreviations as needed

    for token in doc[:-1]:
        # Default assumption: token is not the start of a sentence
        token.is_sent_start = False

        # Check for potential sentence start
        if token.text == '.':
            # Check if the previous token is a two-letter lowercase abbreviation
            if token.i > 0 and doc[token.i - 1].text.lower() in abbreviations:
                doc[token.i + 1].is_sent_start = False
            else:
                # Period that is not after a two-letter lowercase abbreviation
                doc[token.i + 1].is_sent_start = True

    # Explicitly set sentence start for the first token
    doc[0].is_sent_start = True
    return doc


# nlp.add_pipe("is_abbreviation",before="ner")
# nlp.add_pipe("set_abbreviation_boundry",before="ner")
# nlp.add_pipe("ignore_num_as_sentence_boundry",before='ner')
# nlp.add_pipe("is_inside_paranthesis",before='ner')
# nlp.add_pipe("numbered_list_with_space",before='ner')
nlp.add_pipe("custom_boundry",before='ner')

#--------------------------------------------------------------------------------------------------->>
# This will only test 100 brown docs out of 500
total_docs_to_test = 100

# Prediction for the brown corpos
spacy_docs = [nlp(text) for text in brown_natural_docs[:total_docs_to_test]]
spacy_sents_str = [[sents.text_with_ws for sents in doc.sents] for doc in spacy_docs]
spacy_sent_indicies = [sent_indices_from_list(doc, space_included=True) for doc in spacy_sents_str]

# Answer key from the brown corpos
spacy_metrics = np.array([evaluate_indices(brown_sent_indicies[i], spacy_sent_indicies[i])
                   for i in range(total_docs_to_test)])


spacy_metrics_avg = score(*spacy_metrics.sum(axis=0))
print("Precision: %.3f" % spacy_metrics_avg[0])
print("Recall: %.3f" % spacy_metrics_avg[1])
print("F1-score: %.3f" % spacy_metrics_avg[2])

