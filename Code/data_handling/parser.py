from pycorenlp import StanfordCoreNLP
import nltk
import json
from nltk.tree import Tree as Tree

class Parser:
    def __init__(self, coreNLPServer ='http://localhost:9000'):
        self.nlp = StanfordCoreNLP('http://localhost:9000')

    def word_list(self, text):
        nlp_output = self.nlp.annotate(text, properties={
            'annotators': 'tokenize,ssplit',
            'outputFormat': 'json'
        })
        word_array = []
        for sentence in nlp_output['sentences']:
            for w in sentence['tokens']:
                word_array.append(w['word'].lower())
        return word_array


    def parse_tree(self, text, binary=False, preprocessed=False):
        nlp_output = self.nlp.annotate(text, properties={
            'annotators': 'tokenize,ssplit,pos,parse',
            'outputFormat': 'json',
            'parse.binaryTrees': 'true'
        })
        if type(nlp_output) == str:
            nlp_output = json.loads(nlp_output, strict=False)

        parse_tree_array = []
        for s in nlp_output['sentences']:
            p_tree = Tree.fromstring(s['parse'])

            if binary:
                nltk.treetransforms.chomsky_normal_form(p_tree)

            if preprocessed:
                p_tree = preprocess_parse_tree(p_tree)

            parse_tree_array.append(p_tree)

        return parse_tree_array

    def draw_parse_tree(self, parse_tree):
        if isinstance(parse_tree, list):
            for t in parse_tree:
                nltk.draw.tree.draw_trees(t)
        else:
            nltk.draw.tree.draw_trees(parse_tree)

#replaces the subtrees with depth 2 with the respective leave
def preprocess_parse_tree( tree):
    p_tree = nltk.tree.ParentedTree.fromstring(str(tree))
    assert isinstance(tree, Tree)
    all_one_child_nodes_removed = False

    while not all_one_child_nodes_removed:
        all_one_child_nodes_removed = True
        for index in reversed(p_tree.treepositions()):
            if isinstance(p_tree[index], Tree) and p_tree[index].height() == 2 and len(tree[index]) == 1:
                tree[index] = tree[index][0]
                all_one_child_nodes_removed = False
        p_tree = tree
    return tree
