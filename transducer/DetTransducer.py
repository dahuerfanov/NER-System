import numpy as np


class DetTransducer:

    def __init__(self, d, w, rho):
        self.d = d
        self.w = w
        self.rho = rho

    def size(self):
        return len(self.d)

    def applyTransducer(self, text):
        q, j = 0, 0
        out_text = []
        for i in text:
            out_text.extend(self.w[q][i])
            q = self.d[q][i]
        out_text.extend(self.rho[q])
        return out_text

    def __str__(self):
        strTrans = ""
        vst = np.zeros(shape=self.size())
        S = []
        S.append(0)
        vst[0] = 1
        while len(S) > 0:
            u = S.pop()
            strTrans += str(u) + " (" + str(not self.rho[u] is None) + ", " + str(self.rho[u]) + ") = "
            for c in range(len(self.d[u])):
                if self.d[u][c] < 0:
                    continue
                strTrans += "(" + str(self.d[u][c]) + "," + str(c) + "," + str(self.w[u][c]) + "), "
            strTrans += "\n"
            for q in self.d[u]:
                if q >= 0 and vst[q] == 0:
                    vst[q] = 1
                    S.append(q)

        return strTrans
