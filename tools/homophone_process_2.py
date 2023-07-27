import io
import os
import time
import jieba

from tqdm import tqdm

from src.utils import read_file
from src.utils import write_file
from src.utils import write_json

from src.retrieval.pinyin_retriever import PinyinRetriever
# from src.detection.cheat_detector   import CheatDetector

def check_ent(ref, hyp, entities):
    for ent, score in entities:
        if ent in ref and ent not in hyp:
        # if ent in ref:
            return True

def filter_utt(ids, hyps, refs, confuse_ents):
    confuse_hyps = []
    confuse_refs = []
    for id, hyp, ref in tqdm(zip(ids, hyps, refs)):
        if check_ent(ref, hyp, confuse_ents):
            confuse_hyps.append([id, hyp])
            confuse_refs.append([id, ref])
    return confuse_hyps, confuse_refs

def filter_homophone(ent, result):
    homophone = []
    for score, entity in result:
        if entity == ent:
            continue
        if score == 1.0:
            homophone.append(entity)
        elif score < 1.0:
            return homophone

def entity_confusion(entity_sets):
    topk = len(retriever.contexts)
    confuse_set = []
    for entity in tqdm(entity_sets):
        result = retriever.retrieve_one_step('', [entity, '', [0, len(entity)]], topk)
        homophone = filter_homophone(entity, result)
        if len(homophone) > 0:
            confuse_set.append([len(homophone), entity])
    return sorted(confuse_set, reverse=True)

def filter(main_set, sub_set):
    new_set = []
    for ent, score in main_set:
        if ent in sub_set:
            new_set.append([ent, score])
    return new_set

HYP_PATH    = "./datas/aishell_test_set/asr_transcription/conformer/hyp"
REF_PATH    = "./datas/aishell_test_set/ref"

FULL_ENTITY_PATH = "./datas/entities/aishell/all_ctx_entities.txt"
ENTITY_PATH      = "./datas/entities/aishell/test/test_1_entities.txt"
OUTPUT_DIR       = "./datas/aishell_test_set/asr_transcription/conformer/confuse"

hyp_datas = read_file(HYP_PATH, sp=' ')
ref_datas = read_file(REF_PATH, sp=' ')

ids  = [id  for id, hyp in hyp_datas]
hyps = [hyp for id, hyp in hyp_datas]
refs = [ref for id, ref in ref_datas]

retriever = PinyinRetriever(FULL_ENTITY_PATH)

entities = read_file(ENTITY_PATH)
entities = [e[0] for e in entities]

# print(entities)
# datas = entity_confusion(entities)

# datas = [[entity, str(confuse)] for confuse, entity in datas]

# output_path = os.path.join(OUTPUT_DIR, 'homophone_set')
# write_file(output_path, datas)

CONFUSION_PATH = "./datas/aishell_test_set/asr_transcription/conformer/confuse/homophone_set"
confusion_sets = read_file(CONFUSION_PATH, sp=' ')

confusion_sets = [[entity, float(score)] for entity, score in confusion_sets]

confusion_sub_sets = confusion_sets[:100]

confuse_hyps, confuse_refs = filter_utt(ids, hyps, refs, confusion_sub_sets)

print(confuse_hyps)
print(len(confuse_hyps))

hyp_output_path = os.path.join(OUTPUT_DIR, 'hyp_homophone_small_set')
ref_output_path = os.path.join(OUTPUT_DIR, 'ref_homophone_small_set')

write_file(hyp_output_path, confuse_hyps)
write_file(ref_output_path, confuse_refs)