from parser import Parser
import nltk

p = Parser()


def sentiment(text):
    nlp_output = p.nlp.annotate(text, properties={
        'annotators': 'sentiment',
        'outputFormat': 'json'
    })
    return nlp_output['sentences'][0]['sentiment']