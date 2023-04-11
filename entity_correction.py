import os

from src.detection.bert_detector import BertDetector

from src.retrieval.pinyin_retriever import PinyinRetriever

if __name__ == '__main__':
    # detection_model_path = "/share/nas165/amian/experiments/speech/AISHELL-NER/outputs/best_model"
    # detector = BertDetector(detection_model_path)
    
    # texts = [
    #     "北京和张家口要占据着相当明显的优势",
    #     "北京和张家口两地的生产总值是二万二千七百三十点八亿元"
    # ]

    # results = detector.predict(texts)
    # print(results)

    entity_path = "/share/nas165/amian/experiments/speech/AISHELL-NER/dump/2023_27_01__20_46_40/all_entities.json"

    retriever = PinyinRetriever(entity_path)
    
    entities = ["糾結倫", "林拒絕"]
    results = retriever.retrieve(entities)
    print(results)

