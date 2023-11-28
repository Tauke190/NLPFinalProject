import nltk
from nltk.corpus import brown

nltk.download('brown')


for sentence in brown.sents()[:10]:
    sentence_text = ' '.join(sentence)
    print('\n')
    print(sentence_text)

# for paragraph in brown.paras()[:10]:
#     print("Paragraph:")
#     for sentence in paragraph:
#         sentence_text = ' '.join(sentence)
#         print(sentence_text)
#     print("\n")