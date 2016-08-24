#
# Parts of the code from:
# https://github.com/ofirnachum/tree_rnn
#

import numpy as np
import theano
from theano import tensor as T
from theano.compat.python2x import OrderedDict


theano.config.floatX = 'float32'

class Node(object):
    def __init__(self, val=None):
        self.children = []
        self.val = val
        self.idx = None
        self.height = 1
        self.size = 1
        self.num_leaves = 1
        self.parent = None
        self.label = None

    def _update(self):
        self.height = 1 + max([child.height for child in self.children if child] or [0])
        self.size = 1 + sum(child.size for child in self.children if child)
        self.num_leaves = (all(child is None for child in self.children) +
                           sum(child.num_leaves for child in self.children if child))
        if self.parent is not None:
            self.parent._update()

    def add_child(self, child):
        self.children.append(child)
        child.parent = self
        self._update()

    def add_children(self, other_children):
        self.children.extend(other_children)
        for child in other_children:
            child.parent = self
        self._update()

class BinaryNode(Node):
    def __init__(self, val=None):
        super(BinaryNode, self).__init__(val=val)

    def add_left(self, node):
        if not self.children:
            self.children = [None, None]
        self.children[0] = node
        node.parent = self
        self._update()

    def add_right(self, node):
        if not self.children:
            self.children = [None, None]
        self.children[1] = node
        node.parent = self
        self._update()

    def get_left(self):
        if not self.children:
            return None
        return self.children[0]

    def get_right(self):
        if not self.children:
            return None
        return self.children[1]


class RNTN:

    def __init__(self, emb_dim, output_dim, learning_rate=0.01,
                 train_embeddings=True,
                 labels_on_nonroot_nodes=False):

        assert emb_dim > 1 and output_dim >= 1 and learning_rate > 0

        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.train_embeddings = train_embeddings
        self.labels_on_nonroot_nodes = labels_on_nonroot_nodes

        self.params = [] #list of params that shall be trained
        self.embeddings = theano.shared(self.init_matrix([self.num_emb, self.emb_dim]))
        if train_embeddings:
            self.params.append(self.embeddings)

    def weight_init_scale(self):
        # weight initialization parameter initialization scale
        # formula: r = sqrt(6/(fanIn + fanOut))
        return 0.01 * np.sqrt(6/float(2*self.emb_dim))

    def init_tensor(self, shape, scale=None):
        if scale is None:
            scale = self.weight_init_scale()
        return np.random.normal(scale=scale, size=shape).astype(theano.config.floatX)

    def init_vector(self, shape):
        return np.zeros(shape, dtype=theano.config.floatX)

    def create_recursive_unit(self):
        self.V = theano.shared(self.init_tensor([2*self.emb_dim, self.emb_dim, 2*self.emb_dim]))
        self.W = theano.shared(self.init_tensor([self.emb_dim, 2*self.emb_dim]))
        self.b = theano.shared(self.init_vector([self.emb_dim, 1]))
        self.params.extend([self.V, self.W, self.b])

        def unit(child_emb1, child_emb2):  # very simple
            h = T.concatenate([child_emb1,child_emb2])
            p = T.tensordot(T.transpose(h), self.V, axes=1)
            e = T.tensordot(p, h, axes=1)
            f = T.tanh(e + T.dot(self.W, h) + self.b)
            return f
        return unit
