import io
import os
import time
import jieba

from src.utils import read_file
from src.utils import read_json
from src.utils import write_file
from src.utils import write_json

OUTPUT_DIR = './dump'

def filter(A_set, B_set, C_set):
    New_set = (A_set - B_set).union(C_set)
    return sorted(list(New_set))

if __name__ == '__main__':
    ctx_path    = './datas/entities/aishell/descriptions/ctx.json'
    
    all_test_entity_path  = './datas/entities/aishell/test/test_1_entities.txt'
    sub_test_entity_paths = [
        # './datas/entities/aishell/test/test_0.1_entities.txt',
        # './datas/entities/aishell/test/test_0.2_entities.txt',
        # './datas/entities/aishell/test/test_0.02_entities.txt',
        # './datas/entities/aishell/test/test_0.05_entities.txt',
        './datas/entities/aishell/test/test_0_entities.txt',
    ] 

    output_path = './datas/entities/aishell'

    all_test_ent_datas = set(sorted([e[0] for e in read_file(all_test_entity_path, sp=" ")]))
    
    sub_test_ent_datas = {}
    for sub_test_ent_path in sub_test_entity_paths:
        key = sub_test_ent_path.split('_')[1]
        sub_test_ent_datas[key] = set(sorted([
            e[0] for e in read_file(sub_test_ent_path, sp=" ")
        ]))

    datas = read_json(ctx_path)
    full_ent_datas = set(sorted([e['entity'] for e in datas]))

    print(f'full: {len(full_ent_datas)}')

    for key in sub_test_ent_datas:
        result_ent_datas = filter(full_ent_datas, all_test_ent_datas, sub_test_ent_datas[key])
        print(f'test {key}: {len(result_ent_datas)}')
        out_path = os.path.join(output_path, f'all_{key}_entities.txt')

        result_ent_datas = [[d] for d in result_ent_datas]
        write_file(out_path, result_ent_datas)
    # ctx_ent_datas  = [e['entity'] for e in datas]
    # ent_datas      = {e['entity']: e for e in datas}

    # entities = []
    # for ent in test_ent_datas:
    #     if ent in ctx_ent_datas:
    #         entities.append(ent)
    #         print(ent_datas[ent]['intro'])
    #         print('_' * 30)

    # print(len(entities))

    # ctx_ent_datas = [[e['entity']] for e in datas]
    # write_file(output_path, ctx_ent_datas)