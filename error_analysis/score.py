import os
import time
import argparse

from tqdm  import tqdm
from jiwer import cer

from src.utils import read_file
from src.utils import write_file
from src.utils import write_json

from src.detection.cheat_detector import CheatDetector as AlignDetector

OUTPUT_DIR = "./dump"

def remove_entities(text, entities):
    for ent in entities:
        text = text.replace(ent, '')
    return text

def recall(hyp_ents, ref_ents):
    hit = 0
    for hyp_ent, ref_ent in zip(hyp_ents, ref_ents):
        hit += 1 if hyp_ent == ref_ent else 0
    return hit / len(ref_ents)

def get_error_analysis(entity_path, refs, hyps):
    detector = AlignDetector(entity_path)

    ent_refs, ent_hyps         = [], []
    non_ent_refs, non_ent_hyps = [], []
    ent_recall     = 0
    ent_utt_length = 0

    total_cer = cer(refs, hyps)

    for ref, hyp in zip(refs, hyps):
        hyp_prediction, align_datas = detector.predict_one_step(
            ref, hyp, return_align=True
        )
        ref_prediction, align_ref, align_hyp = align_datas

        corrupt_entities, correct_entities = [], []
        for hyp_pred, ref_pred in zip(hyp_prediction, ref_prediction):
            corrupt_entity, _, position = hyp_pred
            correct_entity, _, _        = ref_pred
            ent_start, ent_end = position
            corrupt_entities.append(corrupt_entity)
            correct_entities.append(correct_entity)
        non_ent_ref = remove_entities(ref, correct_entities)
        non_ent_hyp = remove_entities(hyp, corrupt_entities)
        
        if len(non_ent_ref) > 0:
            non_ent_refs.append(non_ent_ref)
            non_ent_hyps.append(non_ent_hyp)

        if len(correct_entities) > 0:
            ent_recall += recall(correct_entities, corrupt_entities)
            ent_utt_length += 1
            ent_ref = "".join(correct_entities)
            ent_hyp = "".join(corrupt_entities)
            ent_refs.append(ent_ref)
            ent_hyps.append(ent_hyp)

    # Entity Recall
    ent_recall /= ent_utt_length
    # Entity CER
    entity_cer = cer(ent_refs, ent_hyps)
    # Non-Entity CER
    non_entity_cer = cer(non_ent_refs, non_ent_hyps)

    result = {
        'cer'           : f'{total_cer * 100:.2f}',
        'entity_recall' : f'{ent_recall * 100:.2f}',
        'entity_cer'    : f'{entity_cer * 100:.2f}',
        'non_entity_cer': f'{non_entity_cer * 100:.2f}'
    }
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity_path",  type=str, required=True)
    parser.add_argument("--ref_path"   ,  type=str, required=True)
    parser.add_argument("--hyp_path"   ,  type=str, required=True)
    args = parser.parse_args()

    refs = [text for id, text in read_file(args.ref_path, sp=" ")]
    hyps = [text for id, text in read_file(args.hyp_path, sp=" ")]

    result   = get_error_analysis(args.entity_path, refs, hyps)
    print(result)

    time_now = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
    exp_dir  = os.path.join(OUTPUT_DIR, time_now)
    os.mkdir(exp_dir)
    print(f'save to {exp_dir}...')

    config_path = os.path.join(exp_dir, 'config.json')
    write_json(config_path, args.__dict__)

    output_path = os.path.join(exp_dir, 'result.json')
    write_json(output_path, result)