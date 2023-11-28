import re
from abbreviations import abbreviations



def replace_str_index(text,index=0,replacement=''):
    return '%s%s%s'%(text[:index],replacement,text[index+1:])

def sentence_segmenter(text):

    indexofperiod = []

    for i in range(len(text)):
        if(text[i] == '.'):
            indexofperiod.append(i)

    # 1 letter before period not empty means maybe . is in the middle name . Replace . with <MIDDLENAME> on the Middle names
    for index in indexofperiod:
        if(text[index-2] == ' '):
           text = replace_str_index(text,index,'<MIDDLENAME>')

    for abbr in abbreviations:
        text = text.replace(abbr, abbr.replace('.', '<PERIOD>'))
    
    print(text)

    # Split sentences using regular expression for '.', '?', and '!'
    sentences = re.split(r'(?<=[.!?])\s*', text)

    # Replace the placeholder back to period in each sentence
    sentences = [s.replace('<PERIOD>', '.') for s in sentences]
    sentences = [s.replace('<MIDDLENAME>', '.') for s in sentences]

    # Filter out empty sentences
    sentences = [s for s in sentences if s]

    return sentences

# Example text
text = "At 5 a.m. Mr. Smith went to the bank. He left the bank at 6 P.M. Mr. Smith then went to the store."

# Segmenting the sentences
segmented_sentences = sentence_segmenter(text)
print(segmented_sentences)

