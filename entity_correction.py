import os
import time
import jieba
import argparse

from tqdm import tqdm

from src.utils import read_file
from src.utils import read_nbest
from src.utils import write_file
from src.utils import write_json

from src.detection.bert_detector   import BertDetector
from src.detection.ckip_detector   import CkipDetector
from src.detection.nbest_detector  import NbestDetector
from src.detection.cheat_detector  import CheatDetector
from src.detection.pinyin_detector import PinyinDetector

from src.retrieval.pinyin_retriever   import PinyinRetriever
from src.retrieval.semantic_retriever import SemanticRetriever
from src.retrieval.prsr_retriever     import PRSRRetriever

from src.rejection.nbest_rejector import NbestRejector

OUTPUT_DIR = './dump'

def NameEntityCorrector(args, texts, detector, retriever, ref_texts=None, nbests=None, nbest_detector=None):
    if args.detection_model_type == "cheat_detector":
        predictions = detector.predict(ref_texts, texts)
    else:
        predictions = detector.predict(texts)
    
    if args.use_rejection:
        predictions_nbest = nbest_detector.predict_no_detect(texts, nbests, predictions)
    final_texts = []
    for i, prediction in tqdm(enumerate(predictions)):
        query_text = [texts[i] for _ in range(len(prediction))]
        results    = retriever.retrieve(query_text, prediction)
        candiates  = [result[0][1] for result in results]

        if args.use_rejection:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr_transcription_path"      ,  type=str, required=True)
    parser.add_argument("--asr_manuscript_path"         ,  type=str, required=True)

    parser.add_argument("--detection_model_type"        ,  type=str, required=True)
    parser.add_argument("--detection_model_path"        ,  type=str, required=True)
    
    parser.add_argument("--retrieval_model_type"        ,  type=str, required=True)
    parser.add_argument("--retrieval_model_path"        ,  type=str, required=False)

    parser.add_argument("--entity_path"                 ,  type=str, required=True)
    parser.add_argument("--entity_content_path"         ,  type=str, required=False)
    parser.add_argument("--entity_vectors_path"         ,  type=str, required=False)

    parser.add_argument("--use_rejection"               ,  type=str, required=True)
    parser.add_argument("--asr_nbest_transcription_path",  type=str, required=False)
    parser.add_argument("--rejection_model_path"        ,  type=str, required=False)
    
    args = parser.parse_args()

    asr_texts = read_file(args.asr_transcription_path, sp=' ')
    indexis   = [data[0] for data in asr_texts]
    texts     = [" ".join(data[1:]) for data in asr_texts]

    ref_texts = read_file(args.asr_manuscript_path, sp=' ')
    ref_texts = [" ".join(data[1:]) for data in ref_texts]
    
    if args.detection_model_type == "bert_detector":
        detector  = BertDetector(args.detection_model_path)
    elif args.detection_model_type == "ckip_detector":
        detector  = CkipDetector(args.detection_model_path)
    elif args.detection_model_type == "cheat_detector":
        detector  = CheatDetector(args.entity_path)
    elif args.detection_model_type == "pinyin_detector":
        detector  = PinyinDetector(args.entity_path)

    if args.retrieval_model_type == "pinyin_retriever":
        retriever = PinyinRetriever(args.entity_path)
    elif args.retrieval_model_type == "semantic_retriever":
        retriever = SemanticRetriever(
            args.retrieval_model_path,
            args.entity_path,
            args.entity_content_path,
            args.entity_vectors_path
        )
    elif args.retrieval_model_type == "prsr_retriever":
        retriever = PRSRRetriever(
            args.retrieval_model_path,
            args.entity_path,
            args.entity_content_path,
            args.entity_vectors_path
        )

    args.use_rejection = True if args.use_rejection == "True" else False

    print(args.use_rejection)

    if not args.use_rejection:
        results = NameEntityCorrector(args, texts, detector, retriever, ref_texts=ref_texts)
    else:
        nbests_dict = read_nbest(args.asr_nbest_transcription_path, sp=' ')
        nbests  = [nbests_dict[index][1:] for index, _ in asr_texts]
        nbest_detector = NbestDetector(model="None")
        
        results = NameEntityCorrector(args, texts, detector, retriever, ref_texts, nbests, nbest_detector)

    time_now = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
    exp_dir  = os.path.join(OUTPUT_DIR, time_now)
    os.mkdir(exp_dir)
    print(f'save to {exp_dir}...')

    hyp = [[index, result] for index, result in zip(indexis, results)]
    ref = [[index, result] for index, result in zip(indexis, ref_texts)]

    config_path = os.path.join(exp_dir, 'config.json')
    write_json(config_path, args.__dict__)

    res_path = os.path.join(exp_dir, 'hyp.txt')
    write_file(res_path, hyp)

    res_path = os.path.join(exp_dir, 'ref.txt')
    write_file(res_path, ref)

    jieba.load_userdict(args.entity_path)

    hyp = [[" ".join(jieba.cut(result, cut_all=False)), f"(aishell_{index})"] for index, result in zip(indexis, results)]
    ref = [[" ".join(jieba.cut(result, cut_all=False)), f"(aishell_{index})"] for index, result in zip(indexis, ref_texts)]

    res_path = os.path.join(exp_dir, 'hyp.trn.txt')
    write_file(res_path, hyp)

    res_path = os.path.join(exp_dir, 'ref.trn.txt')
    write_file(res_path, ref)