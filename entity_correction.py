import os
import time

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
        candiates = NbestRejector.reject(prediction, predictions_nbest[i], candiates)
        
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
    asr_transcription_path       = "/share/nas165/amian/experiments/speech/espnet/workspace/esun_zh_tcpgen/asr1/exp/asr_train_asr_conformer_raw_zh_char_use_wandbtrue_sp/decode_asr_transformer_asr_model_valid.acc.ave_10best/aishell_ner/test/text"
    asr_nbest_transcription_path = "/share/nas165/amian/experiments/speech/espnet/workspace/esun_zh_tcpgen/asr1/exp/asr_train_asr_conformer_raw_zh_char_use_wandbtrue_sp/decode_asr_transformer_asr_model_valid.acc.ave_10best/aishell_ner/test/logdir"

    asr_texts   = read_file(asr_transcription_path, sp=' ')
    nbests_dict = read_nbest(asr_nbest_transcription_path, sp=' ')

    indexis = [index for index, text in asr_texts]
    texts   = [text for index, text in asr_texts]
    nbests  = [nbests_dict[index][1:] for index, _ in asr_texts]


    detection_model_path = "/share/nas165/amian/experiments/speech/AISHELL-NER/outputs/best_model"
    entity_path          = "/share/nas165/amian/experiments/speech/AISHELL-NER/dump/2023_27_01__20_46_40/all_entities.json"

    detector       = BertDetector(detection_model_path)
    nbest_detector = NbestDetector(model=detector.model)
    retriever      = PinyinRetriever(entity_path)

    results = NameEntityCorrector(texts, nbests, detector, nbest_detector, retriever)
    results = [[index, result] for index, result in zip(indexis, results)]

    time_now = time.strftime("%Y_%d_%m__%H_%M_%S", time.localtime())
    exp_dir  = os.path.join(OUTPUT_DIR, time_now)
    os.mkdir(exp_dir)
    print(f'save to {exp_dir}...')

    res_path = os.path.join(exp_dir, 'hyp')
    write_file(res_path, results)