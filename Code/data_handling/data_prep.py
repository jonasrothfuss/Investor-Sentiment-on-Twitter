import nltk
from nltk.tree import Tree
from TreeLSTM import tree_rnn
from data_handling import parser
from data_handling import tweet
from data_handling import db_handling
import pickle
import time

def clean_tweets_in_df(s104dataframe):
    for i in range(len(s104dataframe["tweet"])):
        if i % 20000 == 0:
            print(i)
        cleaned_tweet = tweet.clean_tweet(s104dataframe.ix[i, 'tweet'])
        s104dataframe.set_value(i, 'tweet', cleaned_tweet)
    pickle.dump(s104dataframe, open(db_handling.sentiment104_PATH + 'train_cleaned.pickle', 'wb'))
    return s104dataframe

def normalize_sent_label(s104dataframe):
    label_translation_dict = {0: -1, 2: 0, 4: 1}
    for i in range(len(s104dataframe["label"])):
        if i % 20000 == 0:
            print(i)
        new_label = label_translation_dict[s104dataframe.ix[i, 'label']]
        s104dataframe.set_value(i, 'label', new_label)
    pickle.dump(s104dataframe, open(db_handling.sentiment104_PATH + 'train_cleaned.pickle', 'wb'))
    return s104dataframe

def build_binary_rnn_tree(tree, sent_label = None, subtree_root = None):
    #delete root
    if tree.label() == 'ROOT':
        tree = tree[0]

    if subtree_root:
        root = subtree_root
    else:
        root = tree_rnn.BinaryNode()
        root.label = sent_label
    for t_node in tree:
        node = tree_rnn.BinaryNode()
        root.add_child(node)
        if isinstance(t_node, Tree) and t_node.height() >= 2:
            build_binary_rnn_tree(t_node, subtree_root = node)
        else: #leave node
            node.val = t_node
    return root

def create_vocab(text_array, vocab_file_path = None):
    # extracts all words in an text_array and generates a vocabulary set
    # vocabulary set may be stored in a file

    p = parser.Parser()
    word_buffer = []

    n = 0
    ts = time.clock()
    for text_element in text_array:
        text = tweet.clean_tweet(text_element) #replace entities ushc as url in text
        word_buffer.extend(p.word_list(text))
        if n % 1000 == 0:
            print(str(n) + "   Processing Time: " + str(time.clock()-ts) + ' sec')
            ts = time.clock()
        n += 1

    vocab = set(word_buffer)

    with open(vocab_file_path, 'w') as f:
        f.writelines(map(lambda x: x + "\n", vocab))

    return vocab
