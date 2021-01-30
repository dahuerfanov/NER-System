import numpy as np

from lexicalTagger.trie import Trie
from transducer.transducer_util import local_extension, compose


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

        rules = []
        for C in range(len(self.tag_set)):
            rules.append(Rule([C, 0], 1, 0))
            rules.append(Rule([0, C], 0, 0))

            for D in range(len(self.tag_set)):
                rules.append(Rule([C, D, 0], 2, 0))
                rules.append(Rule([C, 0, D], 1, 0))
                rules.append(Rule([0, C, D], 0, 0))

                for E in range(len(self.tag_set)):
                    rules.append(Rule([C, D, E, 0], 3, 0))
                    rules.append(Rule([C, D, 0, E], 2, 0))
                    rules.append(Rule([C, 0, D, E], 1, 0))
                    rules.append(Rule([0, C, D, E], 0, 0))

                    for F in range(len(self.tag_set)):
                        rules.append(Rule([C, D, E, F, 0], 4, 0))
                        rules.append(Rule([C, D, E, 0, F], 3, 0))
                        rules.append(Rule([C, D, 0, E, F], 2, 0))
                        rules.append(Rule([C, 0, D, E, F], 1, 0))
                        rules.append(Rule([0, C, D, E, F], 0, 0))

        final_trans = []
        for it in range(num_rules):
            err_mat = np.zeros(shape=(len(self.tag_set), len(self.tag_set)), dtype=int)
            for i in range(len(true_tags)):
                if true_tags[i] != lex_tags[i]:
                    err_mat[lex_tags[i]][true_tags[i]] += 1
            tag_from, tag_to = -1, -1
            cnt_err = -1
            for i in range(len(self.tag_set)):
                for j in range(len(self.tag_set)):
                    if cnt_err < err_mat[i][j]:
                        tag_from, tag_to = i, j
                        cnt_err = err_mat[i][j]

            idx_best = -1
            best_score = 0
            tags_best_trans = []
            print("finding best rule #" + str(it) + " out of " + str(len(rules)) + " for " + str(cnt_err) + " mistakes")
            print("from " + self.tag_set[tag_from] + " to " + self.tag_set[tag_to])
            for i, rule in enumerate(rules):
                rule.P[rule.idx] = tag_from
                rule.new_symbol = tag_to

                score_cur_trans, tags_cur_trans = rule.apply(lex_tags, true_tags)
                if score_cur_trans > best_score:
                    best_score = score_cur_trans
                    tags_best_trans = tags_cur_trans
                    idx_best = i

            if idx_best < 0:
                break
            print("best: " + str(rules[idx_best].P) + ", " + str(rules[idx_best].idx) + " " + self.tag_set[
                rules[idx_best].new_symbol])
            print("score: " + str(best_score) + "/" + str(len(lex_tags)))
            final_trans.append(local_extension(rules[idx_best].P, rules[idx_best].idx,
                                               rules[idx_best].new_symbol, len(self.tag_set)))
            lex_tags = tags_best_trans

        t = final_trans[0]
        for i in range(1, len(final_trans), 1):
            t = compose(final_trans[i], t)
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

    def apply(self, tags, true_tags):
        score = 0
        tags2 = []
        q = -1
        for i in range(len(tags)):
            while q >= 0 and self.P[q + 1] != tags[i]:
                q = self.pi[q]
            if self.P[q + 1] == tags[i]:
                q = q + 1
            tags2.append(tags[i])
            if tags[i] == true_tags[i]:
                score += 1
            if q == len(self.P) - 1:
                tags2[i - len(self.P) + 1 + self.idx] = self.new_symbol
                q = -1  # if pattern P is found we forget the acc idx q since we consider just non-overlapping matches
                if tags[i - len(self.P) + 1 + self.idx] == true_tags[i - len(self.P) + 1 + self.idx]:
                    score -= 1
                if self.new_symbol == true_tags[i - len(self.P) + 1 + self.idx]:
                    score += 1

        return score, tags2


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
