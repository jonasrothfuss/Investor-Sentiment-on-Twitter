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

        if len(nlp_output['sentences']) > 1:
            #merge trees from sentences
            tree_string = "(Top "
            for s in nlp_output['sentences']:
                p_tree = Tree.fromstring(s['parse'])
                tree_string += str(p_tree[0])
            tree_string += ")"
            merged_tree = Tree.fromstring(tree_string)
        else:
            #no merging required
            merged_tree = Tree.fromstring(nlp_output['sentences'][0]['parse'])
            #remove root
            merged_tree = merged_tree[0]

        if binary:
            nltk.treetransforms.chomsky_normal_form(merged_tree)

        if preprocessed:
            merged_tree = preprocess_parse_tree(merged_tree)

        return merged_tree

    def draw_parse_tree(self, parse_tree):
        nltk.draw.tree.draw_trees(parse_tree)

#replaces the subtrees with depth 2 with the respective leave
def preprocess_parse_tree(tree):
    assert isinstance(tree, Tree)
    all_one_child_nodes_removed = False

    while not all_one_child_nodes_removed:
        all_one_child_nodes_removed = True
        for index in reversed(tree.treepositions()):
            if tree.height() == 2:
                break
            if isinstance(tree[index], Tree) and tree[index].height() == 2 and len(tree[index]) == 1:
                tree[index] = tree[index][0]
                all_one_child_nodes_removed = False
    return tree
