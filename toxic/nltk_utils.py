import nltk
from nltk.corpus import stopwords
import spacy
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
# Tweet tokenizer does not split at apostophes which is what we want
from nltk.tokenize import TweetTokenizer
import re
from html.parser import HTMLParser

#settings
eng_stopwords = set(stopwords.words("english"))

lem = WordNetLemmatizer()
tokenizer=TweetTokenizer()

import tqdm


def tokenize_sentences(sentences, words_dict):
    tokenized_sentences = []
    for sentence in tqdm.tqdm(sentences):
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


APPO = {
"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" we all",
"didn't": "did not",
"tryin'":"trying"
}


lem = WordNetLemmatizer()
tokenizer=TweetTokenizer()

def cleanhtml(raw_html):
    htmlparser = HTMLParser()
    cleantext = htmlparser.unescape(raw_html)
    # cleanr = re.compile('<.*?>')
    # cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


def remove_punctutation(text_with_punct):
    punctuations = '''!()-[]{};:"\,<>./?@#$%^&*_~'''
    # remove punctuation from the string
    no_punct = ""
    for char in text_with_punct:
        if char not in punctuations:
            no_punct = no_punct + char
    return no_punct


def clean(comment):
    """
    This function receives comments and returns clean word-list
    """
    # remove<U+00B7>
    comment = re.sub("\\n", "", comment)
    comment = re.sub("<.*>", "", comment)
    # Convert to lower case , so that Hi and hi are the same
    comment = comment.lower()
    # remove \n
    comment = re.sub("\\n", "", comment)

    # remove leaky elements like ip,user
    comment = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "", comment)
    #removing usernames
    comment=re.sub("\[\[.*\]","",comment)

    # removing html tags
    comment = cleanhtml(comment)

    # removing punctation
    comment = remove_punctutation(comment)

    # Split the sentences into words
    words = tokenizer.tokenize(comment)

    # (')aphostophe  replacement (ie)   you're --> you are
    # ( basic dictionary lookup : master dictionary present in a hidden block of code)
    words = [APPO[word] if word in APPO else word for word in words]
    words = [lem.lemmatize(word, "v") for word in words]
    words = [w for w in words if not w in eng_stopwords]

    clean_sent = " ".join(words)

    return (clean_sent)