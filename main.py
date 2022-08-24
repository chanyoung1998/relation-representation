from korre.korre import KorRE
from cls_model import *
from relation_extraction_model import predict


korre = KorRE()
relations = {}

sentence = "찬영이에게 사빈이는 눈물을 보였습니다. 찬영이는 그런 사빈이를 위로해줬습니다. 사빈이는 감사한 마음을 담아 찬영이에게 선물을 줬습니다."

for sent in sentence.split('.')[:-1]:
    ner = korre.ner_sub_obj(sent)
    sub_obj = result(ner, emo_predict(sent))

    if f'{sub_obj[0]}-{sub_obj[1]}' not in relations.keys():
        relations[f'{sub_obj[0]}-{sub_obj[1]}'] = sub_obj[2]
    else:
        relations[f'{sub_obj[0]}-{sub_obj[1]}'] += "/" + sub_obj[2]

print(relations)

# ------------------------------------------ #
sent = "찬영이에게 동생, 사빈이는 눈물을 보였습니다"
tagged_sent = korre.ner_tagged(sent)
print(predict(tagged_sent))

#TODO
