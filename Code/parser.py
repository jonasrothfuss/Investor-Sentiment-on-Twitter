from pycorenlp import StanfordCoreNLP
import nltk
import json

class Parser:
    def __init__(self, coreNLPServer ='http://localhost:9000'):
        self.nlp = StanfordCoreNLP('http://localhost:9000')

    def parse_tree(self, text, binary = False):
        nlp_output = self.nlp.annotate(text, properties={
            'annotators': 'tokenize,ssplit,pos,parse',
            'outputFormat': 'json',
            'parse.binaryTrees': 'true'
        })
        if type(nlp_output) == str:
            nlp_output = json.loads(nlp_output, strict=False)

        parse_tree_array = []
        for s in nlp_output['sentences']:
            p_tree = nltk.tree.Tree.fromstring(s['parse'])

            if binary:
                nltk.treetransforms.chomsky_normal_form(p_tree)

            parse_tree_array.append(p_tree)

        return parse_tree_array


    def draw_parse_tree(self, parse_tree):
        if isinstance(parse_tree, list):
            for t in parse_tree:
                nltk.draw.tree.draw_trees(t)
        else:
            nltk.draw.tree.draw_trees(parse_tree)
