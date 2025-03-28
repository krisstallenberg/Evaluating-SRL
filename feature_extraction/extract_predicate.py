import nltk
from nltk.stem import WordNetLemmatizer

def extract_predicate_lemma(sentence):
    lemmatizer = WordNetLemmatizer()
    for word in sentence:
        if word['predicate'] != '_':
            return lemmatizer.lemmatize(word['form'], pos='v')