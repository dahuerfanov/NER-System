import pandas as pd

from brill_ner import BrillNER

path = r"C:\Users\Diego\PycharmProjects\NER-data\ner_dataset.csv"
df = pd.read_csv(path, delimiter=',', encoding="ISO-8859-1")
print(df)

df = df.replace({'Tag': {'B-geo': 'B-loc', 'B-gpe': 'B-org', 'B-tim': 'B-misc', 'B-art': 'B-misc', 'B-eve': 'B-misc',
                         'B-nat': 'B-misc',
                         'I-geo': 'I-loc', 'I-gpe': 'I-org', 'I-tim': 'I-misc', 'I-art': 'I-misc', 'I-eve': 'I-misc',
                         'I-nat': 'I-misc'}})

text = df['Word'].values
tags = df['Tag'].values
print(text)
print(tags)

p_train1 = 0.78
p_train2 = 0.12

text_train1 = text[0:int(len(text) * p_train1)]
tags_train1 = tags[0:int(len(tags) * p_train1)]

text_train2 = text[int(len(text) * p_train1):int(len(text) * (p_train1 + p_train2))]
tags_train2 = tags[int(len(tags) * p_train1):int(len(tags) * (p_train1 + p_train2))]

text_test = text[int(len(text) * (p_train1 + p_train2)):]
tags_test = tags[int(len(tags) * (p_train1 + p_train2)):]

ner = BrillNER(["B-org", "B-per", "B-loc", "B-misc", "I-org", "I-per", "I-loc", "I-misc", "O"])

ner.fit(text_lex=text_train1, tags_lex=tags_train1, text_contex=text_train2, tags_contex=tags_train2,
        num_rules=100, min_prefix=3, max_rule_len=5, out_tag="O")

ner.test(text_test, tags_test)
