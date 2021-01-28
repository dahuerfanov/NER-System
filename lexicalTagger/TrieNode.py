import numpy as np


class TrieNode:

    def __init__(self, n):
        self.isFinal = False
        self.maxIndex = 0
        self.tagCnt = np.zeros(shape=n, dtype=int)
        self.children = dict()
