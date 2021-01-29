import numpy as np

import transducer.Transducer
from lexicalTagger.Trie import Trie
from transducer.Transducer import local_extension


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

    def fit(self, words, tags, words2, tags2, num_rules, min_prefix, out_tag="O"):
        self.out_tag = out_tag
        self.min_prefix = min_prefix
        err_mat = np.zeros(shape=(len(self.tag_set), len(self.tag_set)), dtype=int)
        self.trie = Trie(len(self.tag_set))
        self.trie_prefix = Trie(len(self.tag_set))
        self.trie_canon_form = Trie(len(self.tag_set))

        for i in range(len(words)):
            self.trie.add_word(words[i].lower(), self.tag_idx[tags[i]])
            self.trie_prefix.add_word(words[i], self.tag_idx[tags[i]], min_prefix)
            self.trie_canon_form.add_word(canonical_form(words[i]), self.tag_idx[tags[i]])

        lex_tags = []
        true_tags = []
        for i in range(len(words2)):
            true_tags.append(self.tag_idx[tags2[i]])
            lex_tags.append(self.get_lex_tag(words2[i]))
            if true_tags[-1] != lex_tags[-1]:
                err_mat[true_tags[-1]][lex_tags[-1]] += 1

        rules = []
        for tag_from in range(len(self.tag_set)):
            for tag_to in range(len(self.tag_set)):
                if err_mat[tag_to][tag_from] == 0:
                    continue

                for C in range(len(self.tag_set)):
                    rules.append(Rule([C, tag_from], 1, tag_to))
                    rules.append(Rule([tag_from, C], 0, tag_to))

                    for D in range(len(self.tag_set)):
                        rules.append(Rule([C, D, tag_from], 2, tag_to))
                        rules.append(Rule([C, tag_from, D], 1, tag_to))
                        rules.append(Rule([tag_from, C, D], 0, tag_to))

                        for E in range(len(self.tag_set)):
                            rules.append(Rule([C, D, E, tag_from], 3, tag_to))
                            rules.append(Rule([C, D, tag_from, E], 2, tag_to))
                            rules.append(Rule([C, tag_from, D, E], 1, tag_to))
                            rules.append(Rule([tag_from, C, D, E], 0, tag_to))

                            for F in range(len(self.tag_set)):
                                rules.append(Rule([C, D, E, F, tag_from], 4, tag_to))
                                rules.append(Rule([C, D, E, tag_from, F], 3, tag_to))
                                rules.append(Rule([C, D, tag_from, E, F], 2, tag_to))
                                rules.append(Rule([C, tag_from, D, E, F], 1, tag_to))
                                rules.append(Rule([tag_from, C, D, E, F], 0, tag_to))

        print("# trans: " + str(len(rules)))

        final_trans = []
        used_rule = np.zeros(shape=len(rules), dtype=bool)
        for it in range(num_rules):
            idx_best = -1
            best_score = 0
            tags_best_trans = []
            print("finding best rule #" + str(it))
            for i, rule in enumerate(rules):
                if used_rule[i]:
                    continue
                tags_cur_trans = rule.apply(lex_tags)
                score_cur_trans = compute_score(true_tags, tags_cur_trans)
                if score_cur_trans > best_score:
                    best_score = score_cur_trans
                    tags_best_trans = tags_cur_trans
                    idx_best = i

            if idx_best < 0:
                break
            used_rule[idx_best] = True
            final_trans.append(local_extension(rules[idx_best].P, rules[idx_best].idx,
                                               rules[idx_best].new_symbol, len(self.tag_set)))
            lex_tags = tags_best_trans

        t = final_trans[0]
        for i in range(1, len(final_trans), 1):
            t = transducer.Transducer.compose(final_trans[i], t)
            print("det -> " + str(i))

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

    def test(self, words, tags):
        pred_tags, pred_tags_trans = self.predict(words)
        true_tags = [self.tag_idx[tag] for tag in tags]

        tp, fp, fn = self.get_performance(pred_tags, true_tags)
        print("Performance before rules were applied")
        print("Recall: {recall:.3f}  Precision: {prec:.3f}  F1: {f1:.3f}\n".format(
            recall=(tp / (tp + fn)), prec=(tp / (tp + fp)),
            f1=2. * (tp / (tp + fn)) * (tp / (tp + fp)) / ((tp / (tp + fn)) + (tp / (tp + fp)))))

        tp, fp, fn = self.get_performance(pred_tags_trans, true_tags)
        print("Performance after rules were applied")
        print("Recall: {recall:.3f}  Precision: {prec:.3f}  F1: {f1:.3f}\n".format(
            recall=(tp / (tp + fn)), prec=(tp / (tp + fp)),
            f1=2. * (tp / (tp + fn)) * (tp / (tp + fp)) / ((tp / (tp + fn)) + (tp / (tp + fp)))))

    def get_performance(self, pred_tags, true_tags):
        pred_entities = self.get_entities(pred_tags)
        true_entities = self.get_entities(true_tags)
        tp, fp = 0., 0.
        for p in pred_entities:
            if p in true_entities:
                tp += 1
            else:
                fp += 1
        return tp, fp, len(true_entities) - tp

    def get_entities(self, tags):

        type_, begin_idx = -1, -1
        entities = set()
        for i, tag in enumerate(tags):
            if self.tag_set[tag][0] == 'B':
                if type_ >= 0:
                    entities.add((begin_idx, i - 1, type_))
                type_ = self.types_[self.tag_set[tag][2:]]
                begin_idx = i
            elif self.tag_set[tag][0] == 'I':
                pass
            else:
                if type_ >= 0:
                    entities.add((begin_idx, i - 1, type_))
                type_ = -1

        return entities


class Rule:

    def __init__(self, P, idx, new_symbol):
        self.P = P
        self.idx = idx
        self.new_symbol = new_symbol

        # KMP algorithm for pattern P. We compute the prefix function in pi:
        self.pi = []
        self.pi.append(-1)
        k = -1
        for q in range(1, len(P)):
            while k >= 0 and P[k + 1] != P[q]:
                k = self.pi[k]
            if P[k + 1] == P[q]:
                k = k + 1
            self.pi.append(k)

    def apply(self, tags):
        tags2 = [tag for tag in tags]
        q = -1
        for i in range(len(tags)):
            while q >= 0 and self.P[q + 1] != tags[i]:
                q = self.pi[q]
            if self.P[q + 1] == tags[i]:
                q = q + 1
            if q == len(self.P) - 1:
                tags2[i - len(self.P) + 1 - self.idx] = self.new_symbol
                q = -1  # if pattern P is found we forget the acc idx q since we consider just non-overlapping matches

        return tags2


def compute_score(a, b):
    cnt = 0
    for i in range(len(a)):
        if a[i] == b[i]:
            cnt += 1
    return cnt


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
