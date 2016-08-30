import datetime
import re
from data_handling.tokenizer import preprocess
import string


def stock_symbols(tweet_dict):
    symbols = []
    for s in tweet_dict['entities']['symbols']:
        symbols.append(s['text'])
    return symbols

def clean_tweet(tweet_text, remove_punctuation=True, remove_numbers=True):
    # replace hyperlinks
    tweet_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'url', tweet_text)
    # replace & by 'and'
    tweet_text = tweet_text.replace('&amp;', 'and')
    # remove quotation
    tweet_text = tweet_text.replace('&quot;', '')
    # replace user mentions
    tweet_text = re.sub(r'@[a-zA-Z0-9_]*', 'user', tweet_text)
    if remove_numbers:
        # Remove numbers
        tweet_text = re.sub(r'[0-9]*', '', tweet_text)
    if remove_punctuation:
        # Remove punctuation
        tweet_text = re.sub("[\.]"*3, ' ', tweet_text)
    #remove double spaces
    tweet_text = re.sub(' +', ' ',tweet_text)
    #limit exclamation marks
    tweet_text = re.sub('!![!]*', '!!', tweet_text)

    #Clean hastags
    tweet_text = tweet_text.replace('\#', '')

    # Clean tickers
    tickers = re.findall(r'\$[a-zA-Z0-9]*', tweet_text)
    for t in tickers:
        tweet_text = tweet_text.replace(t, t[1:])

    return tweet_text.lower()



#replaces entities and users in tweet text
def text_w_replaced_entities(tweet_dict):
    text = tweet_dict["text"]
    entities_for_replacement = {}

    #URLS
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for u in urls:
        entities_for_replacement[u] = 'url'

    #Users
    if "user_mentions" in tweet_dict["entities"].keys():
        for u in tweet_dict["entities"]["user_mentions"]:
            entities_for_replacement[(tweet_dict["text"][int(u['indices'][0]):int(u['indices'][1])])] = 'user'
    #Symbols
    if "symbols" in tweet_dict["entities"].keys():
        for u in tweet_dict["entities"]["symbols"]:
            entities_for_replacement[(tweet_dict["text"][int(u['indices'][0]):int(u['indices'][1])])] = tweet_dict["text"][(int(u['indices'][0])+1):int(u['indices'][1])]
    #replace entities
    for entity, replacement in entities_for_replacement.items():
        text = text.replace(entity, replacement)
    return text.lower() #make lower case

def get_lags(stock_symbol):
    pass

class Tweet:
    def __init__(self, tweet_dict):
        self.id = tweet_dict["_id"]
        self.text = clean_tweet(text_w_replaced_entities(tweet_dict))
        self.created_at = datetime.datetime.utcfromtimestamp(int(tweet_dict["timestamp_ms"])//1000)
        self.timestamp = int(tweet_dict["timestamp_ms"])
        self.language = tweet_dict['lang']
        self.retweet_count = tweet_dict["retweet_count"]
        self.urls = tweet_dict['entities']['urls']
        self.hashtags = tweet_dict['entities']['hashtags']
        self.symbols = stock_symbols(tweet_dict)
        self.follower_count = tweet_dict['user']['followers_count']
        self.user = tweet_dict['user']
        self.text_tokens = preprocess(self.text)

    def number_of_urls(self):
        return len(self.urls)

    def number_of_symbols(self):
        return len(self.symbols)

    def has_symbol(self, symbol):
        return symbol.upper() in self.symbols

    def __str__(self):
        s = "Date: " + str(self.created_at) + "\n"
        s += "Text: " + str(self.text) + "\n"
        s += "Symbols: " + str(self.symbols) + "\n"
        return s

