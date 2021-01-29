import numpy as np


class TrieNode:

    def __init__(self, n):
        self.isfinal = False
        self.maxidx = 0
        self.tagcnt = np.zeros(shape=n, dtype=int)
        self.children = dict()
