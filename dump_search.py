import os
import time

from tqdm     import tqdm
from datetime import datetime

from src.utils import read_file
from src.utils import read_json
from src.utils import write_file
from src.utils import write_json

DUMP_DIR = "./dump"

start_date_str = "2023_07_19__19_44_34"[2:]

start_date = datetime.strptime(start_date_str, '%y_%m_%d__%H_%M_%S')

if __name__ == '__main__':
    datas = []
    for folder in sorted(os.listdir(DUMP_DIR)):
        if '_' not in folder:
            continue
        now_date = datetime.strptime(folder[2:], '%y_%m_%d__%H_%M_%S')
        if now_date < start_date:
            continue
        
        exp_dir     = os.path.join(DUMP_DIR, folder)
        config_path = os.path.join(exp_dir, 'config.json')
        config      = read_json(config_path)
        
        entity_path = config['entity_path']
        topk  = config['prsr_topk']
        alpha = config['prsr_alpha']

        result_path = os.path.join(exp_dir, 'result.json')
        result      = read_json(result_path)

        result_key = list(result.keys())
        
        title = ["topk", "alpha"] + result_key
        # title = ["entity_path"] + result_key

        result_value = list(result.values())
        data  = [topk, alpha] + result_value
        
        datas.append(data)

    datas = [title] + datas

    time_now = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
    exp_dir  = os.path.join(DUMP_DIR, time_now)
    os.mkdir(exp_dir)
    print(f'save to {exp_dir}...')

    config_path = os.path.join(exp_dir, 'config.json')
    # write_json(config_path, args.__dict__)
    write_json(config_path, {'start_date': start_date_str})

    analysis_path = os.path.join(exp_dir, 'result.tsv')
    write_file(analysis_path, datas, sp="\t")