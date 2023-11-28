import spacy

correct = 0
answer_key = []
results = []

nlp = spacy.load("en_core_web_sm")
config = {"punct_chars": [".", "?", "!", "。"]}
sentencizer = nlp.add_pipe("sentencizer", config=config)

with open('brown_answer_key.txt','r') as myfile:
    for line in myfile:
        answer_key.append(line)


with open('brown_text.txt','r') as myfile:
    data = myfile.read()


doc = nlp(data)

print(nlp.pipe_names)

for sentence in doc.sents:
    results.append(sentence)

for i in range(len(answer_key)):
    print(results[i],end='')
    print(answer_key[i],end='')
    print('\n')
    correct+=1
    if(results[i].trim('\n') == answer_key[i].trim('\n')):
        correct+=1


print(correct)