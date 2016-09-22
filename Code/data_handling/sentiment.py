import time
import re
import collections
import pandas as pd
import numpy as np


from data_handling.parser import Parser

p = Parser()
label_score_dict = {
    "positive" : 1,
    "neutral" : 0,
    "negative" : -1,
    "verypositive" : 2,
    "verynegative" : -2
}

def sentiment(text):
    nlp_output = p.nlp.annotate(text, properties={
        'annotators': 'sentiment',
        'outputFormat': 'json'
    })
    return nlp_output['sentences'][0]['sentiment']

def sentiment_label_to_score(label):
    return label_score_dict[label.lower()]

def bulk_sentiment(tweets):
    cumulated_score = 0
    sucess_counter = 0
    ts = time.clock()
    for t in tweets:
        try:
            score = sentiment_label_to_score(sentiment(t.text))
            cumulated_score += score
            sucess_counter += 1
            print("--- (" + str(score) + ") --- " + t.text)
            if sucess_counter % 200 == 0:
                duration = time.clock() - ts
                ts = time.clock()
                print("Successfully processed 200 tweets:" + str(sucess_counter) + "  processing time: " + str(duration))
        except:
            print("Sentiment analysis failed on: " + t.text)
    return cumulated_score/float(sucess_counter)

def word_occurrences(tweets_df):
    assert 'labels' in tweets_df.keys() and 'tweet' in tweets_df.keys()


#bag of word model

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)

def preprocess(s, lowercase=True):
    if lowercase:
        s = s.lower()
    tokens = tokenize(s)
    return tokens

def tokenize(s):
    if type(s) is type(''): #string
        return tokens_re.findall(s)
    elif isinstance(s, collections.Iterable):
        tokenized_sentences =[]
        for sentence in s:
            tokenized_sentences.append(tokens_re.findall(sentence))
        return tokenized_sentences

def word_counter(sentences):
    tokens_array = []
    for t in tokenize(sentences):
        tokens_array.extend(t)
    counter = collections.Counter(tokens_array)
    return counter

def counter_per_label(tweets_df):
    assert 'tweet' in tweets_df.keys() and 'label' in tweets_df.keys()
    labels = set(tweets_df['label'])
    counts_dict = {}
    for label in labels:
        tweets = tweets_df[tweets_df['label'] == label]['tweet']
        counts_dict[label] = word_counter(tweets)
    return counts_dict

def word_count_dict_per_label(tweets_df):
    assert 'tweet' in tweets_df.keys() and 'label' in tweets_df.keys()
    words = dict(word_counter(tweets_df['tweet'])).keys()
    counters = counter_per_label(tweets_df)
    #add word with zero count to respective dicts
    counter_dicts = {}
    for label in counters.keys():
        counter_dict = dict(counters[label])
        for w in words:
            if w not in counter_dict.keys():
                counter_dict[w] = 0
        counter_dicts[label] = counter_dict

    #assert that all dicts have the same length
    assert all([len(words)==len(dict)for dict in counter_dicts.values()])

    return counter_dicts

def average_lag_per_word(tweets_df):
    assert 'lag', 'tweet' in tweets_df.keys()
    words = dict(word_counter(tweets_df['tweet'])).keys()
    avg_lag_df = pd.DataFrame(data=np.zeros([len(words), 2]), index=words, columns=['count', 'avg_lag'])
    for i, tweet, lag in zip(range(tweets_df.shape[0]), tweets_df['tweet'], tweets_df['lag']):
        for word in tokenize(tweet):
            print(i)
            old_count = avg_lag_df.loc[word, 'count']
            old_avg_lag = avg_lag_df.loc[word, 'avg_lag']
            new_count = old_avg_lag * (old_count/(old_count+1)) + lag / (old_count+1)

            avg_lag_df.set_value(index=word, col='count', value=old_count + 1)
            avg_lag_df.set_value(index=word, col='avg_lag', value=new_count)

    return avg_lag_df

def word_count_tuples(tweets_df):
    counter_dicts = word_count_dict_per_label(tweets_df)
    labels = counter_dicts.keys()
    words = dict(word_counter(tweets_df['tweet'])).keys()
    word_count_tuple_dict = {}
    for w in words:
        tuple = [counter_dicts[label][w] for label in labels]
        assert len(tuple) == len(labels) and all(count >= 0 for count in tuple)
        word_count_tuple_dict[w] = tuple
    assert len(word_count_tuple_dict) == len(words)
    return word_count_tuple_dict



