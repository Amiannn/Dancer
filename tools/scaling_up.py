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
import random

from tqdm import tqdm

from src.utils import read_file
from src.utils import read_json
from src.utils import write_file
from src.utils import write_json

from src.retrieval.pinyin_retriever import PinyinRetriever

OUTPUT_DIR  = "./datas/entities/aishell/scales"

TEST_ENTITY_PATH  = "./datas/entities/aishell/test/test_1_entities.txt"
TRAIN_ENTITY_PATH = "./datas/entities/aishell/all_0_entities.txt"

test_entities = read_file(TEST_ENTITY_PATH)
test_entities = [e[0] for e in test_entities][:1000]

train_entities = read_file(TRAIN_ENTITY_PATH)
train_entities = [e[0] for e in train_entities]

for size in range(0, 15000, 1000):
    random.shuffle(train_entities)
    entities = test_entities + train_entities[:size]
    print(f'size of the entities: {len(entities)}')
    
    entities_data = [[e] for e in entities]
    output_path = os.path.join(OUTPUT_DIR, f'all_{size + 1000}_scale_entities.txt')
    write_file(output_path, entities_data)
        
