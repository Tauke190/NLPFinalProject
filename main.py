import spacy;
import re

nlp = spacy.load("en_core_web_sm")
boundary = re.compile('^[0-9]$')



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

# Customer boundry rule , semicolon at end of sentence
@spacy.Language.component("semicolon_split")
def set_custom_boundries(doc):
    for token in doc[:-1]:
        if token.text == ';':
            doc[token.i+1].is_sent_start = True
    return doc

nlp.add_pipe("semicolon_split",before='parser')
nlp.add_pipe("numbered_list_with_space",before='parser')

# The new pipeline of spacy 
# print(nlp.pipe_names)

with open('training.txt','r') as myfile:
    data = myfile.read()

# data = "My name is Jonas E. Smith said he couldnt come"


doc = nlp(data)

for sent in doc.sents:
    token = [token.text for token in sent]
    sentences = sent
    print(sent)





