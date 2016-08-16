from pycorenlp import StanfordCoreNLP
import nltk

class Parser:
    def __init__(self, coreNLPServer ='http://localhost:9000'):
        self.nlp = StanfordCoreNLP('http://localhost:9000')

    def parse_tree(self, text):
        nlp_output = self.nlp.annotate(text, properties={
            'annotators': 'tokenize,ssplit,parse',
            'outputFormat': 'json'
        })
        parse_tree_array = []
        for s in nlp_output['sentences']:
            parse_tree_array.append(nltk.tree.Tree.fromstring(s['parse']))
        return parse_tree_array

    def draw_parse_tree(parse_tree):
        if isinstance(parse_tree, list):
            for t in parse_tree:
                nltk.draw.tree.draw_trees(t)
        else:
            nltk.draw.tree.draw_trees(parse_tree)
