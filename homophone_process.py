import io
import os
import time
import jieba

from src.utils import read_file
from src.utils import write_file
from src.utils import write_json

from src.retrieval.pinyin_retriever import PinyinRetriever
from src.detection.cheat_detector   import CheatDetector

OUTPUT_DIR = './dump'

HYP_PATH    = "./datas/aishell_test_set/asr_transcription/conformer/hyp"
REF_PATH    = "./datas/aishell_test_set/ref"
ENTITY_PATH = "./datas/entities/aishell/test_1_entities.txt"

def check_homophone(target, prediction, _id):
    for span in prediction:
        entity, type, position = span
        result = retriever.retrieve_one_step("", span)
        candidate = []
        if result[1] == entity:
            return
        for score, ent in result:
            candidate.append([score, ent])
        
        homophone = []
        top_1 = candidate[0][0]
        for score, ent in candidate:
            if score == top_1:
                homophone.append([score, ent])
        if len(homophone) > 1:
            print(f'prediction: {prediction}')
            print(f'uid:{_id}, entity: {entity}, homophone: {homophone}')
            print("_" * 30)

detector  = CheatDetector(ENTITY_PATH)
retriever = PinyinRetriever(ENTITY_PATH)

entities = read_file(ENTITY_PATH)
entities = [e[0] for e in entities]

hyp_datas = read_file(HYP_PATH, sp=' ')
ref_datas = read_file(REF_PATH, sp=' ')

for hyp_data, ref_data in zip(hyp_datas, ref_datas):
    _id    = hyp_data[0]
    text   = hyp_data[1]
    target = ref_data[1]
    
    prediction = detector.predict_one_step(target, text)
    if len(prediction) == 0:
        continue
    check_homophone(target, prediction, _id)

entity = "佩戴还"
type   = ""
pos    = [0, len(entity) + 1]
span   = [entity, type, pos]
result = retriever.retrieve_one_step("", span)
print(result)

entity = "条带环"
type   = ""
pos    = [0, len(entity) + 1]
span   = [entity, type, pos]
result = retriever.retrieve_one_step("", span)
print(result)

entity = "而在淮"
type   = ""
pos    = [0, len(entity) + 1]
span   = [entity, type, pos]
result = retriever.retrieve_one_step("", span)
print(result)
