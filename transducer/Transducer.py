import numpy as np

from transducer.DetTransducer import DetTransducer
from transducer.Transition import Transition


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


def is_suffix(P, i, j):  # i <= j
    k1 = j
    for k2 in range(i, -1, -1):
        if P[k1] != P[k2]:
            return False
        k1 -= 1
    return True


def is_prefix(P, Q):
    if len(Q) < len(P):
        return False
    for i in range(len(P)):
        if P[i] != Q[i]:
            return False
    return True


def add_joker_transition(transitions, q, alphSize):
    used = np.zeros(shape=alphSize, dtype=int)
    for t in transitions:
        used[t.i] = 1
    for i in range(alphSize):
        if used[i] == 0:
            transitions.append(Transition(i, i, q))


def local_extension(P, k, c, alphSize) -> Transducer:
    adj = []
    F = set()
    m = len(P)

    for q in range(m):
        adj.append([])
        for a in range(alphSize):
            j = q + 1
            while P[j - 1] != a or not is_suffix(P, j - 2, q - 1):
                j -= 1
                if j == 0: break
            adj[q].append(Transition(a, a, j))
        F.add(q)

    sink = m
    adj.append([])
    if k == m - 1:
        q_t = 0
    else:
        q_t = m + 1

    for i in range(k + 1, m, 1):
        q = m + i - k
        adj.append([])
        adj[q].append(Transition(P[i], P[i], (q + 1) % (2 * m - k)))
        add_joker_transition(adj[q], sink, alphSize)

    adj[k].append(Transition(P[k], c, q_t))
    for i in range(k + 1, m, 1):
        S = P[k: i] + P[k:]
        if is_suffix(P, k - 1, i - 1) and not is_prefix(S[0: m - k], S[i - k:]):
            adj[i].append(Transition(P[k], c, q_t))

    return Transducer(adj, F, alphSize)


def compose(t1, t2) -> Transducer:  # t1 o t2
    assert t1.alphsize == t2.alphsize
    n = 0
    m = np.zeros(shape=(t1.size(), t2.size()), dtype=int)
    Q_x = np.zeros(shape=t1.size() * t2.size(), dtype=int)
    Q_y = np.zeros(shape=t1.size() * t2.size(), dtype=int)
    adj = []
    F = set()
    Q_x[n], Q_y[n] = 0, 0  # initial state
    n += 1
    m[0][0] = n
    adj.append([])
    i = 0
    while i < n:
        u_x, u_y = Q_x[i], Q_y[i]
        idx = m[u_x][u_y] - 1
        if u_x in t1.F and u_y in t2.F:
            F.add(idx)
        for tr2 in t2.adj[u_y]:
            for tr1 in t1.adj[u_x]:
                if tr2.o == tr1.i:
                    v_x, v_y = tr1.q, tr2.q
                    if m[v_x][v_y] == 0:
                        adj.append([])
                        Q_x[n], Q_y[n] = v_x, v_y
                        n += 1
                        m[v_x][v_y] = n
                    adj[idx].append(Transition(tr2.i, tr1.o, m[v_x][v_y] - 1))

        i += 1
    return Transducer(adj, F, t1.alphsize)
