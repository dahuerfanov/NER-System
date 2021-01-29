from lexicalTagger.TrieNode import TrieNode


class Trie:

    def __init__(self, tagscnt):
        self.tagscnt = tagscnt
        self.root = TrieNode(tagscnt)
        self.maxlen = 0

    def add_word(self, w, tag, min_prefix=-1):
        if min_prefix < 0:
            min_prefix = len(w) - 1
        node = self.root
        self.maxlen = max(len(w), self.maxlen)
        for i in range(len(w)):
            if not w[i] in node.children:
                node.children[w[i]] = TrieNode(self.tagscnt)
            node = node.children[w[i]]
            if i >= min_prefix:
                node.isfinal = True
                node.tagcnt[tag] += 1
                if node.tagcnt[tag] > node.tagcnt[node.maxidx]:
                    node.maxidx = tag

    def get_tag(self, w, min_prefix=-1):
        if min_prefix < 0:
            min_prefix = len(w) - 1
        node = self.root
        for i in range(len(w)):
            if w[i] in node.children:
                node = node.children[w[i]]
            else:
                return -1
            if i >= min_prefix:
                if node.isfinal:
                    return node.maxidx
        return -1
