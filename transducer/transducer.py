import numpy as np

from transducer.det_transducer import DetTransducer


class Transducer:

    def __init__(self, adj, F, alphSize):

        self.adj = adj
        self.F = F
        self.alphsize = alphSize

    def __str__(self):

        strTrans = ""
        vst = np.zeros(shape=len(self.adj), dtype=int)
        S = []
        S.append(0)
        vst[0] = 1
        while len(S) > 0:
            u = S.pop()
            if u in self.F:
                strInF = "True"
            else:
                strInF = "False"
            strTrans += str(u) + " (" + strInF + ") = "
            for t in self.adj[u]:
                strTrans += "(" + str(t.q) + "," + str(t.i) + "," + str(t.o) + "), "
            strTrans += "\n"
            for t in self.adj[u]:
                if vst[t.q] == 0:
                    vst[t.q] = 1
                    S.append(t.q)
        return strTrans

    def size(self):
        return len(self.adj)

    def determinize(self) -> DetTransducer:
        d = []
        w_det = []
        rho = []
        C = []
        S = set()
        S.add((0, ()))
        C.append(S)
        n = 1
        w_length = [0] * self.alphsize
        q = 0
        while q < n:
            S = C[q]
            d.append([])
            w_det.append([])
            rho.append(None)

            w = [None] * self.alphsize
            S_ = [None] * self.alphsize
            for (q_, u) in S:
                if q_ in self.F:
                    if not rho[q] is None:
                        assert is_same_list(u, rho[q])  # condition satisfied by finite deterministic transducers
                    else:
                        rho[q] = u

                for t in self.adj[q_]:
                    if w[t.i] is None:
                        w[t.i] = list(u)
                        w[t.i].append(t.o)
                        w_length[t.i] = len(w[t.i])
                    else:
                        w_length[t.i] = max_prefix(w[t.i], u, t.o, w_length[t.i])

            for (q_, u) in S:
                for t in self.adj[q_]:
                    if S_[t.i] is None:
                        S_[t.i] = set()
                    S_[t.i].add((t.q, tuple(inverse(w[t.i], w_length[t.i], u, t.o))))

            for j in range(self.alphsize):
                e = -1
                for i in range(n):
                    if C[i] == S_[j]:
                        e = i
                        break
                if e < 0:
                    e = len(C)
                    C.append(S_[j])
                    n += 1
                d[q].append(e)
                w_det[q].append(w[j][0: w_length[j]])
            q += 1

        return DetTransducer(d, w_det, rho)


def is_same_list(a, b):
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def inverse(w, prefix, u, ucont):
    u_ = list(u)
    u_.append(ucont)
    for i in range(prefix):
        if w[i] != u_[i]:
            return []
    return u_[prefix:]


def max_prefix(s1, s2, s2next, acc):
    i = 0
    while i < len(s1) and i <= len(s2) and i < acc:
        if i < len(s2):
            if s1[i] != s2[i]: break
        else:
            if s1[i] != s2next: break
        i += 1
    return i
