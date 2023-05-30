import os
import time
import argparse

from tqdm import tqdm

from src.utils import read_file
from src.utils import write_file
from src.utils import write_json

from src.detection.bert_detector   import BertDetector
from src.detection.ckip_detector   import CkipDetector
from src.detection.nbest_detector  import NbestDetector
from src.detection.cheat_detector  import CheatDetector
from src.detection.pinyin_detector import PinyinDetector

OUTPUT_DIR = "./dump"

def get_accuracy(refs, hyps):
    if len(refs) == 0 and len(hyps) == 0:
        return 1
    elif len(hyps) == 0:
        return 0
    length = len(hyps)
    count  = 0
    for hyp in hyps:
        if hyp in refs:
            count += 1
    return count / length

def get_recall(refs, hyps):
    if len(refs) == 0:
        return 1
    length = len(refs)
    count  = 0
    for ref in refs:
        if ref in hyps:
            count += 1
    return count / length

def check_cross(a, b_datas):
    a_list = set(range(*a))
    for b in b_datas:
        b_list = set(range(*b))
        if len(b_list - a_list) != len(b_list):
            return True
    return False

def get_error_analysis(refs, hyps):
    diff_ref = refs.copy()
    for ref in refs:
        if ref in hyps:
            hyps.remove(ref)
            diff_ref.remove(ref)
    refs = diff_ref
    if len(hyps) == 0:
        return None
    # hyp
    cross = 0
    for hyp in hyps:
        if check_cross(hyp, refs):
            cross += 1
    cross = cross / len(hyps)
    no_cross = 1 - cross
    return cross, no_cross

def get_position(datas):
    results = []
    for data in datas:
        result = []
        for hyp in data:
            result.append(hyp[-1])
        results.append(result)
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity_path"         ,  type=str, required=True)
    parser.add_argument("--ref_path"            ,  type=str, required=True)
    parser.add_argument("--hyp_path"            ,  type=str, required=True)
    parser.add_argument("--model_type"          ,  type=str, required=True)
    parser.add_argument("--model_path",  type=str, required=False)
    args = parser.parse_args()

    gt_detector  = CheatDetector(args.entity_path)

    if args.model_type   == "bert_detector":
        detector  = BertDetector(args.model_path)
    elif args.model_type == "ckip_detector":
        detector  = CkipDetector(args.model_path)
    elif args.model_type == "pinyin_detector":
        detector  = PinyinDetector(args.entity_path)

    ref_texts = read_file(args.ref_path, sp=' ')
    ref_texts = [" ".join(data[1:]) for data in ref_texts]
    hyp_texts = read_file(args.hyp_path, sp=' ')
    hyp_texts = [" ".join(data[1:]) for data in hyp_texts]

    # get groundtruth
    ref_detection_result = gt_detector.predict(ref_texts, hyp_texts)
    # detector predict
    hyp_detection_result = detector.predict(hyp_texts)
    
    # get entity position
    ref_detection_result = get_position(ref_detection_result)
    hyp_detection_result = get_position(hyp_detection_result)

    recall, accuracy = 0, 0
    cross, no_cross  = 0, 0
    error_length     = 0
    length           = 0

    for ref_detect, hyp_detect in zip(ref_detection_result, hyp_detection_result):
        recall   += get_recall(ref_detect, hyp_detect)
        accuracy += get_accuracy(ref_detect, hyp_detect)
        length   += 1
        err_res = get_error_analysis(ref_detect, hyp_detect)
        if err_res == None:
            continue
        cross        += err_res[0]
        no_cross     += err_res[1]
        error_length += 1

    recall   /= length
    accuracy /= length
    f1 = 2 * ((recall * accuracy) / (recall + accuracy))


    cross    = cross / error_length if error_length != 0 else 0
    no_cross = no_cross / error_length if error_length != 0 else 0

    print(f'recall  : {recall:.4f}')
    print(f'accuracy: {accuracy:.4f}')
    print(f'f1      : {f1:.4f}')
    
    print(f'cross   : {cross:.4f}')
    print(f'no_cross: {no_cross:.4f}')

    time_now = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
    exp_dir  = os.path.join(OUTPUT_DIR, time_now)
    os.mkdir(exp_dir)
    print(f'save to {exp_dir}...')

    config_path = os.path.join(exp_dir, 'config.json')
    write_json(config_path, args.__dict__)

    result_path = os.path.join(exp_dir, 'result.json')
    write_json(result_path, {
        'recall': recall,
        'accuracy': accuracy,
        'f1': f1,
        'cross': cross,
        'no_cross': no_cross,
    })
