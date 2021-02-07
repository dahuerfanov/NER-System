import numpy as np


class Trie:

    def __init__(self, tagscnt):
        self.tagscnt = tagscnt
        self.root = TrieNode(tagscnt)

    def add_word(self, w, tag, min_prefix=0):
        if min_prefix <= 0:
            min_prefix = len(w)
        node = self.root
        for i in range(len(w)):
            if not w[i] in node.children:
                node.children[w[i]] = TrieNode(self.tagscnt)
            node = node.children[w[i]]
            if i >= min_prefix - 1:
                node.isfinal = True
                node.tagcnt[tag] += 1
                if node.tagcnt[tag] > node.tagcnt[node.maxidx]:
                    node.maxidx = tag

    def get_tag(self, w, min_prefix=0):
        if min_prefix <= 0:
            min_prefix = len(w)
        node = self.root
        for i in range(len(w)):
            if w[i] in node.children:
                node = node.children[w[i]]
            else:
                return -1
            if i >= min_prefix - 1:
                if node.isfinal:
                    return node.maxidx
        return -1


class TrieNode:

    def __init__(self, n):
        self.isfinal = False
        self.maxidx = 0
        self.tagcnt = np.zeros(shape=n, dtype=int)
        self.children = dict()
