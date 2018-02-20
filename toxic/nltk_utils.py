import nltk
import tqdm
from toxic.spell_correct import correct_sentence

from nltk.corpus import stopwords

def tokenize_sentences(sentences, words_dict):
    stopWords = set(stopwords.words('english'))
    tokenized_sentences = []
    for sentence in tqdm.tqdm(sentences):
        sentence = correct_sentence(sentence)
        if hasattr(sentence, "decode"):
            sentence = sentence.decode("utf-8")
        tokens = nltk.tokenize.word_tokenize(sentence)
        result = []
        for word in tokens:
            word = word.lower()
            if word not in words_dict:
                words_dict[word] = len(words_dict)
            word_index = words_dict[word]
            result.append(word_index)
        tokenized_sentences.append(result)
    return tokenized_sentences, words_dict


def correct_sentence(sentence):
    return correct_sentence(sentence)