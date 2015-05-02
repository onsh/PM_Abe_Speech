import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from replacers import RegexpReplacer
import pandas as pd
from pandas import Series

def opener(o):
    with open(o, 'rt') as f:
        return f.read()


def tokenizer(raw):
    # print(stopwords.words('english'))
    stop_words = stopwords.words('english')
    symbols = ["'", '"', '`', '.', ',', '-', '!', '?', ':', ';',
               '(', ')', '--', '\'s', '\'', '\'re', '{', '}', 'ー']

    replacer = RegexpReplacer()
    replaced_raw = replacer.replace(raw).lower()
    tokens = [word 
                 for word in word_tokenize(replaced_raw)
                     if word not in stop_words + symbols]

    text = nltk.Text(replaced_raw)
    return (tokens, text)


def stem(word):
    regexp = r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$'
    stem, suffix = re.findall(regexp, word)[0]
    return stem


def findtags(tag_prefix, tagged_text):
    cfd = nltk.ConditionalFreqDist((tag, word) for (word, tag) in tagged_text
                                      if tag.startswith(tag_prefix))
    return dict((tag, cfd[tag].most_common(10)) for tag in cfd.conditions())


def output(tokens):
    obj = Series(tokens)
    top10 = obj.value_counts()[:10]
    print(top10)
    
    top10_list = list(top10.keys())
    text.dispersion_plot(top10_list)
    

speech = opener('PM_Abe_speech.txt')
statement = opener('US-JPN_Joint_Vison_Statement.txt')

speech_tokens, speech_text = tokenizer(speech)
print("Length of PM Abe's Speech:", len(speech_text), 'words')
output(speech_tokens)

statement_tokens, statement_text = tokenizer(statement)
print('Length of US-JPN Statement:', len(statement_text), 'words')
output(statement_tokens)

tagged_speech_tokens = nltk.pos_tag(speech_tokens)
tagdict = findtags('NN', tagged_speech_tokens)
for tag in sorted(tagdict):
    print(tag, tagdict[tag])

