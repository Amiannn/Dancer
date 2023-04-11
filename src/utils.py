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

def read_json(path):
    with open(path, 'r', encoding='utf-8') as fr:
        return json.load(fr)

def write_file(path, datas):
    with open(path, 'w', encoding='utf-8') as fr:
        for data in datas:
            fr.write(" ".join(data) + '\n')
