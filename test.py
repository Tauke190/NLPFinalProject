import spacy

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")

@spacy.Language.component("set_custom_boundaries")
def set_custom_boundaries(doc):
    for token in doc[:-1]:
        if token.text == '...':
            doc[token.i+1].is_sent_start = True
    return doc

# Add the custom boundary function to the pipeline
nlp.add_pipe("set_custom_boundaries", before='parser')

# Process the text
sentence = '@Jessica_Chobot did you see the yakuza vs zombies....smh but cool at the same time'
doc = nlp(sentence)

# Segment the text into sentences
sentences = [sent.text for sent in doc.sents]
print(sentences)
