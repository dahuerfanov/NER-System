from contextual_tagger.aho_corasick import AhoCorasickAutomaton
from contextual_tagger.transducer_util import local_extension, compose
from lexical_tagger.lexical_tagger import LexicalTagger


class BrillNER:

    def __init__(self, tag_set, default_tag):
        self.tag_set = tag_set
        self.tag_idx = dict()
        self.trie = None
        self.trie_prefix = None
        self.trie_canon_form = None
        self.trans = None
        self.default_tag = default_tag
        for i in range(len(self.tag_set)):
            self.tag_idx[self.tag_set[i]] = i

    def fit(self, text_lex, tags_lex, text_contex, tags_contex, num_rules, min_prefix, max_rule_len):

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
        '''

        # Build the trie structures from the text and tags for the lexical tagger
        self.lex_tagger = LexicalTagger(self.tag_set, min_prefix, self.tag_idx[self.default_tag])
        for word, tag in zip(text_lex, tags_lex):
            self.lex_tagger.add_word(word, self.tag_idx[tag])

        # Initialize the tags for contextual tagger according to the lexical tagger
        lex_tags = []
        true_tags = []
        for word, tag in zip(text_contex, tags_contex):
            true_tags.append(self.tag_idx[tag])
            lex_tags.append(self.lex_tagger.predict(word))

        # Generate templates of all possible contextual rules of max. length max_rule_len
        aho_coras_autom = AhoCorasickAutomaton(len(self.tag_set), max_rule_len)
        print("size aho-automaton: " + str(aho_coras_autom.n))

        # Find the best num_rules rules or halt before finding them according to the alpha param.
        final_trans = []
        for it in range(num_rules):
            print("finding best rule #" + str(it))
            best_score, best_rule = aho_coras_autom.match(lex_tags, true_tags, len(self.tag_set),
                                                          self.simple_transformation_fun)
            print("best: " + str([self.tag_set[tag] for tag in best_rule.P]) + ", " + str(best_rule.idx) +
                  " to " + self.tag_set[best_rule.new_symbol])
            print("score: " + str(best_score))
            if best_score <= 0:
                break
            best_score, lex_tags = best_rule.apply(lex_tags, true_tags, self.simple_transformation_fun)
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

    def predict(self, words):
        pred_tags = []
        for word in words:
            pred_tags.append(self.lex_tagger.predict(word))

        return pred_tags, self.trans.apply(pred_tags)

    def simple_transformation_fun(self, a, symbol):
        return symbol
