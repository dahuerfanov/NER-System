from lexical_tagger.trie import Trie


class LexicalTagger():

    def __init__(self, tag_set, min_prefix, default_tag):
        self.tag_set = tag_set
        self.min_prefix = min_prefix
        self.default_tag = default_tag
        self.trie = Trie(len(self.tag_set))
        self.trie_canon_form = Trie(len(self.tag_set))

    def add_word(self, word, tag):
        self.trie.add_word(word.lower(), tag, self.min_prefix)
        self.trie_canon_form.add_word(canonical_form(word), tag)

    def predict(self, word):
        tag = self.trie.get_tag(word.lower())
        if tag < 0:
            tag = self.trie.get_tag(word.lower(), self.min_prefix)
        if tag < 0:
            tag = self.trie_canon_form.get_tag(canonical_form(word))
        if tag < 0:
            tag = self.trie_canon_form.get_tag(canonical_form(word), self.min_prefix)
        if tag < 0:
            tag = self.default_tag
        return tag


def canonical_form(s):
    c_form = ""
    for c in s:
        if c.isupper():
            c_form += "X"
        elif c.islower():
            c_form += "x"
        elif c.isdigit():
            c_form += "d"
        else:
            c_form += "p"

    return c_form
