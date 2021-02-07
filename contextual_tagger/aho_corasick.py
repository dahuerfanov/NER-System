import numpy as np


class AhoCorasickAutomaton:

    def __init__(self, in_alph_size, max_length_rules):
        self.in_alph_size = in_alph_size
        self.max_length_rules = max_length_rules
        self.n = 0
        self.next = []
        self.leaf = []
        self.parent = []
        self.pi = []
        self.pi_dict = []
        self.dict = []
        # create root
        self.add_node(-1)

        v = 0
        acc_word = [[]]
        while v < len(acc_word):
            for symbol in range(in_alph_size):
                self.next[v][symbol] = self.n
                new_word = acc_word[v] + [symbol]
                self.add_node(v)
                if len(new_word) > 1:
                    self.leaf[-1] = True
                self.dict[-1] = new_word
                if len(new_word) < max_length_rules:
                    acc_word.append(new_word)
            v += 1

    def add_node(self, parent):
        self.next.append([-1] * self.in_alph_size)
        self.leaf.append(False)
        self.parent.append(parent)
        self.pi.append(-1)
        self.pi_dict.append(-1)
        self.dict.append(None)
        self.n += 1

    def get_link(self, v):
        if self.pi[v] < 0:
            # root or a child of the root
            if self.parent[v] <= 0:
                self.pi[v] = 0
            else:
                u = self.get_link(self.parent[v])
                while u > 0 > self.next[u][self.dict[v][-1]]:
                    u = self.get_link(u)

                self.pi[v] = max(0, self.next[u][self.dict[v][-1]])

        return self.pi[v]

    def get_dict_link(self, v):
        if self.pi_dict[v] < 0 < self.parent[v]:
            u = self.get_link(v)
            while u > 0 and not self.leaf[u]:
                u = self.get_link(u)
            self.pi_dict[v] = u

        return self.pi_dict[v]

    def match(self, text, true_text, out_alph_size, transformation_fun):
        score = np.zeros(shape=(self.n, self.max_length_rules, out_alph_size), dtype=int)
        v = 0
        best_score = -int(1e9)
        best_rule_P = None
        best_rule_idx = None
        best_rule_new_sym = None
        for i in range(len(text)):
            while v > 0 > self.next[v][text[i]]:
                v = self.get_link(v)
            if self.next[v][text[i]] > 0:
                v = self.next[v][text[i]]
            u = v
            while u > 0:
                # match found
                if self.leaf[u]:
                    # changing symbol index for resp. rule
                    for idx in range(len(self.dict[u])):
                        idx_text = i - len(self.dict[u]) + 1 + idx
                        for symbol in range(out_alph_size):
                            trans_symbol = transformation_fun(self.dict[u][idx], symbol)
                            if trans_symbol == self.dict[u][idx]:
                                continue
                            if text[idx_text] == true_text[idx_text]:
                                score[u][idx][symbol] -= 1
                            if true_text[idx_text] == trans_symbol:
                                score[u][idx][symbol] += 1
                            if best_score < score[u][idx][symbol]:
                                best_score = score[u][idx][symbol]
                                best_rule_P = self.dict[u]
                                best_rule_idx = idx
                                best_rule_new_sym = symbol

                u = self.get_dict_link(u)

        return best_score, Rule(best_rule_P, best_rule_idx, best_rule_new_sym)


class Rule:

    def __init__(self, P, idx, new_symbol):
        self.P = P
        self.idx = idx
        self.new_symbol = new_symbol

    def apply(self, tags, true_tags, transformation_fun):
        # KMP algorithm for pattern P. We compute the prefix function in pi
        pi = []
        pi.append(-1)
        k = -1
        for q in range(1, len(self.P)):
            while k >= 0 and self.P[k + 1] != self.P[q]:
                k = pi[k]
            if self.P[k + 1] == self.P[q]:
                k = k + 1
            pi.append(k)

        score = 0
        tags2 = []
        q = -1
        for i in range(len(tags)):
            while q >= 0 and self.P[q + 1] != tags[i]:
                q = pi[q]
            if self.P[q + 1] == tags[i]:
                q = q + 1
            tags2.append(tags[i])
            if q == len(self.P) - 1:
                trans_symbol = transformation_fun(tags[i - len(self.P) + 1 + self.idx], self.new_symbol)
                tags2[i - len(self.P) + 1 + self.idx] = trans_symbol
                # If pattern P is found we forget the acc idx q since we consider just non-overlapping matches
                q = -1
                if tags[i - len(self.P) + 1 + self.idx] == true_tags[i - len(self.P) + 1 + self.idx]:
                    score -= 1
                if trans_symbol == true_tags[i - len(self.P) + 1 + self.idx]:
                    score += 1

        return score, tags2
