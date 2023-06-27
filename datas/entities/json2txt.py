import os

from src.utils import read_file
from src.utils import read_json
from src.utils import write_file

input_path  = "./blists/aishell"
output_path = "./blists/aishell"

def remove_short(datas):
    result = []
    for data in datas:
        if len(data) < 2:
            continue
        result.append(data)
    return result

for filename in os.listdir(input_path):
    if '.json' not in filename:
        continue
    path  = os.path.join(input_path, filename)
    datas = read_json(path)
    _tmp = []
    for t in datas:
        _tmp.extend([rare.upper() for rare in datas[t]])
    datas = sorted(list(set(_tmp)))
    datas = remove_short(datas)
    datas = [[data] for data in datas]

    output_data_path = os.path.join(output_path, filename.replace('.json', '.txt'))
    write_file(output_data_path, datas)