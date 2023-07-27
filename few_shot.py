"""
count: 0, entity: 615
count: 1, entity: 711
count: 5, entity: 854
count: 10, entity: 932
count: 20, entity: 1000
count: 100, entity: 1130
"""

import io
import os
import time
import jieba

from tqdm import tqdm

from src.utils import read_file
from src.utils import read_json
from src.utils import write_file
from src.utils import write_json

from src.retrieval.pinyin_retriever import PinyinRetriever
# from src.detection.cheat_detector   import CheatDetector

def count_train_entity(detector, refs):
    table = {}
    for ref in tqdm(refs):
        prediction = detector.predict_one_step(ref, ref)
        entities   = [pred[0] for pred in prediction]
        for entity in entities:
            table[entity] = table[entity] + 1 if entity in table else 1
    return table

OUTPUT_DIR  = "./datas/entities/aishell/shots"
REF_PATH    = "/share/nas165/amian/experiments/speech/AISHELL-NER/data/aishell_ner_transcript.train.txt"

TEST_ENTITY_PATH  = "./datas/entities/aishell/test/test_1_entities.txt"
TRAIN_ENTITY_PATH = "./datas/entities/aishell/all_0_entities.txt"

test_entities = read_file(TEST_ENTITY_PATH)
test_entities = [e[0] for e in test_entities]

# ref_datas = read_file(REF_PATH, sp=' ')
# refs = [d[1] for d in ref_datas]

# detector  = CheatDetector(TEST_ENTITY_PATH)

# table = count_train_entity(detector, refs)
# write_json('./count_table.json', table)
table = read_json('./count_table.json')

shots = {i: [] for i in range(102)}
for test_ent in test_entities:
    if test_ent in table:
        count = table[test_ent]
        if count > 100:
            shots[101].append(test_ent)
        else:
            shots[count].append(test_ent)
    else:
        count = 0
        shots[count].append(test_ent)

account = [[] for i in range(102)]
for count in shots:
    account[count] = shots[count]
    if count > 0:
        account[count].extend(account[count - 1])

for count in [0, 1, 5, 10, 20, 100]:
    print(f'count: {count}, entity: {len(account[count])}')
    entities = account[count]
    entities_data = [[e] for e in entities]

    output_path = os.path.join(OUTPUT_DIR, f'test_{count}_shot_entities.txt')
    write_file(output_path, entities_data)
    
