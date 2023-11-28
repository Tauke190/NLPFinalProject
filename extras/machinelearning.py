import spacy
from spacy.gold import GoldParse
from spacy.language import EntityRecognizer

nlp = spacy.load("en_core_web_sm")

text = "This is the first sentence. This is the second sentence. And this is the third sentence."

sentences = ["This is the first sentence.", "This is the second sentence.", "And this is the third sentence."]

annotations = [{"entities": [(0, 25)]}, {"entities": [(0, 26)]}, {"entities": [(0, 25)]}]

train_data = list(zip(sentences, annotations))

nlp.entity.add_label("SENTENCE")

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.begin_training()

    for i in range(10):
        losses = {}

        for text, annotations in train_data:
            doc = nlp.make_doc(text)
            gold = GoldParse(doc, entities=annotations)
            nlp.update([gold], sgd=optimizer, losses=losses)

        print(losses)

doc = nlp(text)

sentences = [sent.text for sent in doc.sents]

for sentence in sentences:
    print(sentence)