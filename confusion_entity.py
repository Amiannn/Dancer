import io
import os
import time
import jieba

from src.utils import read_file
from src.utils import write_file
from src.utils import write_json

from src.retrieval.pinyin_retriever import PinyinRetriever
from src.detection.cheat_detector   import CheatDetector

HYP_PATH    = "./datas/aishell_test_set/asr_transcription/conformer/hyp"
REF_PATH    = "./datas/aishell_test_set/ref"
ENTITY_PATH = "./datas/entities/aishell/test_ctx_entities.txt"

OUTPUT_DIR  = "./datas/aishell_test_set/asr_transcription/conformer/confuse"

def check_confusion_phone(target, prediction, _id):
    uid_datas = []
    for span in prediction:
        entity, type, position = span
        result = retriever.retrieve_one_step("", span)
        candidate = []
        if result[0][1] == entity:
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
            return True
    return False

detector  = CheatDetector(ENTITY_PATH)
retriever = PinyinRetriever(ENTITY_PATH)

entities = read_file(ENTITY_PATH)
entities = [e[0] for e in entities]

hyp_datas = read_file(HYP_PATH, sp=' ')
ref_datas = read_file(REF_PATH, sp=' ')

id2hyp = {d[0]: d for d in hyp_datas}
id2ref = {d[0]: d for d in ref_datas}

uids = []
for hyp_data, ref_data in zip(hyp_datas, ref_datas):
    _id    = hyp_data[0]
    text   = hyp_data[1]
    target = ref_data[1]
    
    prediction = detector.predict_one_step(target, text)
    if len(prediction) == 0:
        continue
    if check_confusion_phone(target, prediction, _id):
        uids.append(_id)

print(uids)

hyp_phonetic_confusion_set = []
ref_phonetic_confusion_set = []

for uid in uids:
    hyp_phonetic_confusion_set.append(id2hyp[uid])
    ref_phonetic_confusion_set.append(id2ref[uid])

    hyp_output_path = os.path.join(OUTPUT_DIR, 'hyp_phonetic_confuse_set')
    ref_output_path = os.path.join(OUTPUT_DIR, 'ref_phonetic_confuse_set')

    write_file(hyp_output_path, hyp_phonetic_confusion_set)
    write_file(ref_output_path, ref_phonetic_confusion_set)