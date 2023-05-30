import os
import json
import time

from tqdm import tqdm

def read_file(path, sp='\t'):
    datas = []
    with open(path, 'r', encoding='utf-8') as frs:
        for fr in frs:
            data = fr.replace('\n', '')
            data = data.split(sp)
            datas.append(data)
    return datas

def write_file(path, datas, sp=" "):
    with open(path, 'w', encoding='utf-8') as fr:
        for data in datas:
            fr.write(sp.join(data) + '\n')

def read_json(path):
    with open(path, 'r', encoding='utf-8') as fr:
        return json.load(fr)

def write_json(path, datas):
    with open(path, 'w', encoding='utf-8') as fr:
        json.dump(datas, fr, indent=4, ensure_ascii=False)

def read_nbest(path, sp=' '):
    # load asr nbest result
    nbest_split_paths = []
    for hyp_split in sorted(os.listdir(path)):
        if 'output.' not in hyp_split: continue
        hyp_split_path = os.path.join(path, hyp_split)
        nbest_len = len(os.listdir(hyp_split_path))
        nbest_split_path = []
        for nbest in range(1, nbest_len + 1):
            hyp_nbest_path = os.path.join(hyp_split_path, f'{nbest}best_recog/text')
            nbest_split_path.append(hyp_nbest_path)
        nbest_split_paths.append(nbest_split_path)

    hyp_dicts = {}
    for nbest_split_path in nbest_split_paths:
        for i in range(len(nbest_split_path)):
            _hyp_datas = read_file(nbest_split_path[i], sp=sp)
            _hyp_datas = [[data[0], " ".join(data[1:])] for data in _hyp_datas]
            for idx, hyp in _hyp_datas:
                hyp_dicts[idx] = hyp_dicts[idx] + [hyp] if idx in hyp_dicts else [hyp]
    return hyp_dicts