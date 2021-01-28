from lexicalTagger.TrieNode import TrieNode


class Trie:

    def __init__(self, numberTags):
        self.numberTags = numberTags
        self.root = TrieNode(numberTags)
        self.maxLen = 0

    def addWord(self, w, tag, min_prefix=-1):
        if min_prefix < 0:
            min_prefix = len(w) -1
        node = self.root
        self.maxLen = max(len(w), self.maxLen)
        for i in range(len(w)):
            if not w[i] in node.children:
                node.children[w[i]] = TrieNode(self.numberTags)
            node = node.children[w[i]]
            if i>=min_prefix:
                node.isFinal = True
                node.tagCnt[tag] += 1
                if node.tagCnt[tag] > node.tagCnt[node.maxIndex]:
                    node.maxIndex = tag

    def getTag(self, w):
        node = self.root
        for i in range(len(w)):
            if w[i] in node.children:
                node = node.children[w[i]]
            else:
                return -1
        if not node.isFinal:
            return -1
        return node.maxIndex

    def hasTag(self, w, tag):
        node = self.root
        for i in range(len(w)):
            if w[i] in node.children:
                node = node.children[w[i]]
            else:
                return True
        if not node.isFinal:
            return True
        return node.tagCnt[tag] > 0