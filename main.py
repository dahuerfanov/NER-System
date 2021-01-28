
import pandas as pd

from BrillNER import BrillNER

path = r"C:\Users\Diego\PycharmProjects\NER-data\ner_dataset.csv"
df = pd.read_csv(path, delimiter=',', encoding = "ISO-8859-1")
print(df)

text = df['Word'].values
tags = df['Tag'].values
print(text)
print(tags)

p_train1 = 0.8
p_train2 = 0.1

text_train1 = text[0:int(len(text)*p_train1)]
tags_train1 = tags[0:int(len(tags)*p_train1)]

text_train2 = text[int(len(text)*p_train1):int(len(text)*(p_train1+p_train2))]
tags_train2 = tags[int(len(tags)*p_train1):int(len(tags)*(p_train1+p_train2))]

text_test = text[int(len(text)*(p_train1+p_train2)):]
tags_test = tags[int(len(tags)*(p_train1+p_train2)):]


ner = BrillNER(["B-geo", "B-org", "B-per", "B-gpe", "B-tim", "B-art",
                         "B-eve", "B-nat", "I-geo", "I-org", "I-per", "I-gpe",
                         "I-tim", "I-art", "I-eve", "I-nat", "O"])

ner.fit(text_train1, tags_train1, text_train2, tags_train2, 10, 4)

ner.test(text_test, tags_test)


