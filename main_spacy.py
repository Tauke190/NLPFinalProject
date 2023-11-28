import spacy;
import re

nlp = spacy.load("en_core_web_sm")
boundary = re.compile('^[0-9]$')
sentences = []
nlp.disable_pipes('parser')


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


nlp.add_pipe("is_abbreviation",before="ner")
# nlp.add_pipe("set_abbreviation_boundry",before="ner")
nlp.add_pipe("ignore_num_as_sentence_boundry",before='ner')
nlp.add_pipe("custom_boundry",before='ner')
nlp.add_pipe("is_inside_paranthesis",before='ner')
nlp.add_pipe("numbered_list_with_space",before='ner')



with open('training.txt','r') as myfile:
    data = myfile.read()

doc = nlp(data)

for sent in doc.sents:
    sentences.append(sent.text)

print(sentences)