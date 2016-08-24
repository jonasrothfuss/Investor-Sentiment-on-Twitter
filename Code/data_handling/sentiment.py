import time

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