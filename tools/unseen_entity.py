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

# unseen
UNSEEN_PATH = "/share/nas167/bicheng/espnet/egs/aishell/mwer/data/statis2/et_unseen"
# rare
RARE_PATH   = "/share/nas167/bicheng/espnet/egs/aishell/mwer/data/statis2/et_rare"
# few
FEW_PATH    = "/share/nas167/bicheng/espnet/egs/aishell/mwer/data/statis2/et_few"

OUTPUT_DIR  = "./datas/entities/aishell"

def filter(entities):
    new_entities = []
    for entity in entities:
        if len(entity) > 1:
            new_entities.append(entity)
    return new_entities

if __name__ == '__main__':
    unseen_entities = filter([e[0] for e in read_file(UNSEEN_PATH, sp=" ")])
    rare_entities   = filter([e[0] for e in read_file(RARE_PATH, sp=" ")])
    few_entities    = filter([e[0] for e in read_file(FEW_PATH, sp=" ")])

    unseen_path = os.path.join(OUTPUT_DIR, 'unseen_entities')
    write_file(unseen_path, [[d] for d in unseen_entities])

    rare_path   = os.path.join(OUTPUT_DIR, 'rare_entities')
    write_file(rare_path, [[d] for d in rare_entities])

    few_path    = os.path.join(OUTPUT_DIR, 'few_entities')
    write_file(few_path, [[d] for d in few_entities])
