import numpy as np

from transducer.transducer import Transducer


# Loal extension algorithm for a contextual rule (https://arxiv.org/abs/2006.11548)
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


# t1 o t2
def compose(t1, t2) -> Transducer:
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


class Transition:

    def __init__(self, i, o, q):
        self.i = i
        self.o = o
        self.q = q


def add_joker_transition(transitions, q, alphSize):
    used = np.zeros(shape=alphSize, dtype=int)
    for t in transitions:
        used[t.i] = 1
    for i in range(alphSize):
        if used[i] == 0:
            transitions.append(Transition(i, i, q))


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
