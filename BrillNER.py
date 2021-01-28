import numpy as np

import transducer.Transducer
from transducer.Transducer import localExtension
from lexicalTagger.Trie import Trie


class BrillNER:

    def __init__(self, tag_set):
        self.tag_set = tag_set
        self.tag_idx = dict()
        self.trie = None
        self.trie_prefix = None
        self.trans = None
        self.out_tag = "O"
        self.types_ = dict()
        cnt_types = 0
        for i in range(len(self.tag_set)):
            self.tag_idx[self.tag_set[i]] = i
            if '-' in self.tag_set[i]:
                if not self.tag_set[i][2:] in self.types_:
                    self.types_[self.tag_set[i][2:]] = cnt_types
                    cnt_types += 1

    def fit(self, words, tags, words2, tags2, num_rules, min_prefix, out_tag="O"):
        self.out_tag = out_tag
        err_mat = np.zeros(shape=(len(self.tag_set), len(self.tag_set)), dtype=int)
        trie = Trie(len(self.tag_set))
        trie_prefix = Trie(len(self.tag_set))

        for i in range(len(words)):
            trie.addWord(words[i].lower(), self.tag_idx[tags[i]])
            for j in range(min_prefix, len(words[i])):
                trie_prefix.addWord(words[i], self.tag_idx[tags[i]], min_prefix)

        lex_tags = []
        true_tags = []
        for i in range(len(words2)):
            true_tags.append(self.tag_idx[tags2[i]])
            lex_tags.append(max(trie.getTag(words2[i]), trie_prefix.getTag(words2[i])))
            if lex_tags[-1] < 0:
                lex_tags[-1] = self.tag_idx[out_tag]
            if true_tags[-1] != lex_tags[-1]:
                err_mat[true_tags[-1]][lex_tags[-1]] += 1

        transducers = []

        for tag_from in range(len(self.tag_set)):
            for tag_to in range(len(self.tag_set)):
                if err_mat[tag_to][tag_from] == 0:
                    continue
                for C in range(len(self.tag_set)):
                    transducers.append(localExtension([C, tag_from], 1, tag_to, len(self.tag_set)))
                    transducers.append(localExtension([tag_from, C], 0, tag_to, len(self.tag_set)))

                    for D in range(len(self.tag_set)):
                        transducers.append(localExtension([C, D, tag_from], 2, tag_to, len(self.tag_set)))
                        transducers.append(localExtension([C, tag_from, D], 1, tag_to, len(self.tag_set)))
                        transducers.append(localExtension([tag_from, C, D], 0, tag_to, len(self.tag_set)))

                        #for E in range(len(self.tag_set)):
                        #    transducers.append(
                        #        localExtension([C, D, E, tag_from], 3, tag_to, len(self.tag_set)))
                        #    transducers.append(
                        #        localExtension([C, D, tag_from, E], 2, tag_to, len(self.tag_set)))
                        #    transducers.append(
                        #        localExtension([C, tag_from, D, E], 1, tag_to, len(self.tag_set)))
                        #    transducers.append(
                        #        localExtension([tag_from, C, D, E], 0, tag_to, len(self.tag_set)))

        print("# trans: " + str(len(transducers)))

        det_transducers = []
        for trans in transducers:
            det_transducers.append(trans.determinize())

        final_trans = []
        used_trans = np.zeros(shape=len(transducers), dtype=bool)
        for _ in range(num_rules):
            idx_best = -1
            best_score = 0
            tags_best_trans = []
            print("finding best trans")
            for i, trans in enumerate(det_transducers):
                if used_trans[i]:
                    continue
                tags_cur_trans = trans.apply(lex_tags)
                score_cur_trans = computeScore(true_tags, tags_cur_trans)
                if score_cur_trans > best_score:
                    best_score = score_cur_trans
                    tags_best_trans = tags_cur_trans
                    idx_best = i

            if idx_best < 0:
                break
            used_trans[idx_best] = True
            final_trans.append(transducers[idx_best])
            lex_tags = tags_best_trans

        t = final_trans[0]
        for i in range(1, len(final_trans), 1):
            t = transducer.Transducer.compose(final_trans[i], t)
            print("i -> " + str(i))

        self.trie = trie
        self.trie_prefix = trie_prefix
        self.trans = t.determinize()

    def predict(self, words):
        pred_tags = []
        for i in range(len(words)):
            pred_tags.append(max(self.trie.getTag(words[i]), self.trie_prefix.getTag(words[i])))
            if pred_tags[-1] < 0:
                pred_tags[-1] = self.tag_idx[self.out_tag]

        return self.trans.apply(pred_tags)

    def test(self, words, tags):
        pred_tags = self.predict(words)
        true_tags = [self.tag_idx[tag] for tag in tags]

        tp, fp, fn = self.getPerformance(pred_tags, true_tags)

        print("Recall: {recall:.3f}  Precision: {prec:.3f}  F1: {f1:.3f}\n".format(
            recall=(tp / (tp + fn)), prec=(tp / (tp + fp)),
            f1=2. * (tp / (tp + fn)) * (tp / (tp + fp)) / ((tp / (tp + fn)) + (tp / (tp + fp)))))

    def getPerformance(self, pred_tags, true_tags):
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


def computeScore(a, b):
    cnt = 0
    for i in range(len(a)):
        if a[i] == b[i]:
            cnt += 1
    return cnt
