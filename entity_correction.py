import os
import time
import argparse

from tqdm import tqdm

from src.utils import read_file
from src.utils import read_nbest
from src.utils import write_file

from src.detection.bert_detector  import BertDetector
from src.detection.nbest_detector import NbestDetector

from src.retrieval.pinyin_retriever import PinyinRetriever

from src.rejection.nbest_rejector import NbestRejector

OUTPUT_DIR = './dump'

def NameEntityCorrector(texts, nbests, detector, nbest_detector, retriever):
    predictions       = detector.predict(texts)
    predictions_nbest = nbest_detector.predict_no_detect(texts, nbests, predictions)

    final_texts = []
    for i, prediction in tqdm(enumerate(predictions)):
        entities  = [entity for entity, entity_type, position in prediction]
        results   = retriever.retrieve(entities)
        candiates = [result[0][1] for result in results]

        print(f'entity   : {entities}')
        print(f'candiates: {candiates}')
        candiates = NbestRejector.reject(prediction, predictions_nbest[i], candiates)

        print(f'final candiates: {candiates}')
        print('_' * 30)
        now, final_text = 0, []
        for candiate, predict in zip(candiates, prediction):
            _, _, position = predict
            start, end = position
            final_text += f'{texts[i][now:start]}{candiate}'
            now = end
        final_text += texts[i][now:]
        final_texts.append("".join(final_text))
    return final_texts

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr_transcription_path"      ,  type=str, required=True)
    parser.add_argument("--asr_nbest_transcription_path",  type=str, required=False)
    parser.add_argument("--detection_model_path"        ,  type=str, required=True)
    parser.add_argument("--entity_path"                 ,  type=str, required=True)
    args = parser.parse_args()

    asr_texts   = read_file(args.asr_transcription_path, sp=' ')
    nbests_dict = read_nbest(args.asr_nbest_transcription_path, sp=' ')

    indexis = [data[0] for data in asr_texts]
    texts   = [" ".join(data[1:]) for data in asr_texts]
    nbests  = [nbests_dict[index][1:] for index, _ in asr_texts]

    detector       = BertDetector(args.detection_model_path)
    nbest_detector = NbestDetector(model=detector.model)
    retriever      = PinyinRetriever(args.entity_path)

    results = NameEntityCorrector(texts, nbests, detector, nbest_detector, retriever)
    results = [[index, result] for index, result in zip(indexis, results)]

    time_now = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
    exp_dir  = os.path.join(OUTPUT_DIR, time_now)
    os.mkdir(exp_dir)
    print(f'save to {exp_dir}...')

    res_path = os.path.join(exp_dir, 'hyp')
    write_file(res_path, results)