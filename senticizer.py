import spacy

nlp = spacy.load("en_core_web_sm")
config = {"punct_chars": [".", "?", "!", "。"]}
sentencizer = nlp.add_pipe("sentencizer", config=config)

text = "This is the first sentence. This is the second sentence. And this is the third sentence."
doc = nlp(text)
sentencizer.to_disk("sentencizer.json")

print(nlp.pipe_names)

for sentence in doc.sents:
    print(sentence.text)