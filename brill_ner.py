from contextual_tagger.aho_corasick import AhoCorasickAutomaton
from contextual_tagger.transducer_util import local_extension, compose
from lexical_tagger.trie import Trie


class BrillNER:

    def __init__(self, tag_set):
        self.tag_set = tag_set
        self.tag_idx = dict()
        self.trie = None
        self.trie_prefix = None
        self.trie_canon_form = None
        self.trans = None
        self.out_tag = "O"
        self.types_ = dict()
        self.min_prefix = -1
        cnt_types = 0
        for i in range(len(self.tag_set)):
            self.tag_idx[self.tag_set[i]] = i
            if '-' in self.tag_set[i]:
                if not self.tag_set[i][2:] in self.types_:
                    self.types_[self.tag_set[i][2:]] = cnt_types
                    cnt_types += 1

    def fit(self, text_lex, tags_lex, text_contex, tags_contex, num_rules, min_prefix, max_rule_len, out_tag="O"):

        '''

        Parameters:
        ----------

        :param text_lex: list                  list of tokens to train the lexical tagger

        :param tags_lex: list                  list of the corresponding tags in text_lex. They must lie in self.tag_set

        :param text_contex: list               list of tokens to train the contextual tagger

        :param tags_contex: list               list of the corresponding tags in text_contex. They must lie in self.tag_set

        :param num_rules: int                  number of contextual rules to learn

        :param min_prefix: int                 minimum length of prefixes to be saved in the prefix trie, used for the
                                               assignation of lexical tags

        :param max_rule_len: int               maximum length of each contextual rule to learn, including the changing
                                               tag

        :param alpha: float                    percentage of minimum amount of mistakes of assigning a particular wrong
                                               tag to another specific one with respect to the total, when considering a
                                               candidate contextual rule

        :param out_tag: string                 the default tag indicating that a token is part of no nominal entity
        '''

        self.out_tag = out_tag
        self.min_prefix = min_prefix
        self.trie = Trie(len(self.tag_set))
        self.trie_prefix = Trie(len(self.tag_set))
        self.trie_canon_form = Trie(len(self.tag_set))

        # Build the trie structures from the text and tags for the lexical tagger
        for i in range(len(text_lex)):
            self.trie.add_word(text_lex[i].lower(), self.tag_idx[tags_lex[i]])
            self.trie_prefix.add_word(text_lex[i], self.tag_idx[tags_lex[i]], min_prefix)
            self.trie_canon_form.add_word(canonical_form(text_lex[i]), self.tag_idx[tags_lex[i]])

        # Initialize the tags for contextual tagger according to the lexical tagger
        lex_tags = []
        true_tags = []
        for i in range(len(text_contex)):
            true_tags.append(self.tag_idx[tags_contex[i]])
            lex_tags.append(self.get_lex_tag(text_contex[i]))

        # Generate templates of all possible contextual rules of max. length max_rule_len
        aho_coras_autom = AhoCorasickAutomaton(len(self.tag_set), max_rule_len)
        print("size aho-automaton: " + str(aho_coras_autom.n))

        # Find the best num_rules rules or halt before finding them according to the alpha param.
        final_trans = []
        for it in range(num_rules):
            print("finding best rule #" + str(it))
            best_score, best_rule = aho_coras_autom.match(lex_tags, true_tags)
            print("best: " + str([self.tag_set[tag] for tag in best_rule.P]) + ", " + str(best_rule.idx) +
                  " to " + self.tag_set[best_rule.new_symbol])
            print("score: " + str(best_score))
            if best_score <= 0:
                break
            best_score, lex_tags = best_rule.apply(lex_tags, true_tags)
            print("best score KMP: " + str(best_score))
            if best_score < 0:
                break
            # Create the contextual_tagger of the best found rule
            final_trans.append(local_extension(best_rule.P, best_rule.idx, best_rule.new_symbol, len(self.tag_set)))

        # Apply the composition operator to the list of transducers in the same order
        t = final_trans[0]
        for i in range(1, len(final_trans), 1):
            t = compose(final_trans[i], t)

        # Finally, determinize the learned contextual_tagger
        self.trans = t.determinize()

    def get_lex_tag(self, word):
        tag = self.trie.get_tag(word)
        if tag < 0:
            tag = self.trie_prefix.get_tag(word)
        if tag < 0:
            tag = self.trie_canon_form.get_tag(canonical_form(word), self.min_prefix)
        if tag < 0:
            tag = self.tag_idx[self.out_tag]
        return tag

    def predict(self, words):
        pred_tags = []
        for i in range(len(words)):
            pred_tags.append(self.get_lex_tag(words[i]))

        return pred_tags, self.trans.apply(pred_tags)


# For lexical tagger
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
            c_form += c

    return c_form
