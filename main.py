import pandas as pd
from sklearn_crfsuite.metrics import flat_classification_report

from brill_ner import BrillNER


class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


path = r"C:\Users\Diego\PycharmProjects\NER-data\ner_dataset.csv"
df = pd.read_csv(path, delimiter=',', encoding="latin1")
df = df.fillna(method="ffill")

text = df['Word'].values
tags = df['Tag'].values
p_train1 = 0.7
p_train2 = 0.2

text_train1 = text[0:int(len(text) * p_train1)]
tags_train1 = tags[0:int(len(tags) * p_train1)]

text_train2 = text[int(len(text) * p_train1):int(len(text) * (p_train1 + p_train2))]
tags_train2 = tags[int(len(tags) * p_train1):int(len(tags) * (p_train1 + p_train2))]

text_test = text[int(len(text) * (p_train1 + p_train2)):]
tags_test = tags[int(len(tags) * (p_train1 + p_train2)):]

ner = BrillNER(["B-geo", "B-org", "B-per", "B-gpe", "B-tim", "B-art", "B-eve", "B-nat", "I-geo", "I-org", "I-per",
                "I-gpe", "I-tim", "I-art", "I-eve", "I-nat", "O"])

ner.fit(text_lex=text_train1, tags_lex=tags_train1, text_contex=text_train2, tags_contex=tags_train2,
        num_rules=30, min_prefix=3, max_rule_len=5, out_tag="O")

getter = SentenceGetter(df[int(len(tags) * (p_train1 + p_train2)):])
sentences = getter.sentences
x = [sent2tokens(s) for s in sentences]
y = [sent2labels(s) for s in sentences]

preds0 = [[ner.tag_set[tag] for tag in ner.predict(sen)[0]] for sen in x]
preds = [[ner.tag_set[tag] for tag in ner.predict(sen)[1]] for sen in x]

report_before = flat_classification_report(y_pred=preds0, y_true=y)
print("Report before rules")
print(report_before)

report_after = flat_classification_report(y_pred=preds, y_true=y)
print("Report after rules")
print(report_after)
