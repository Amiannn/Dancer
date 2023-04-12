from typing import List

from simpletransformers.ner import NERArgs
from simpletransformers.ner import NERModel

from src.detection.abs_detector import AbsDetector

from ckip_transformers.nlp import CkipNerChunker

class CkipDetector(AbsDetector):
    def __init__(self, model_path, use_cuda=True):
        self.ner_driver = CkipNerChunker(model=model_path)

    def _postprocess(self, datas):
        result = []
        for data in datas:
            entity = data.word
            entity_type = data.ner
            start, end = data.idx
            if entity_type not in ['GPE', 'ORG', 'LOC', 'PERSON']:
                continue
            result.append([entity, entity_type, [start, end]])
        return result

    def predict(self, texts: List[str]) -> List[str]:
        predictions = self.ner_driver(texts, use_delim=False)
        predictions = [self._postprocess(pred) for pred in predictions]
        return predictions

    def predict_one_step(self, text: str) -> List[str]:
        return self.predict([text])[0]

if __name__ == "__main__":
    detector = CkipDetector(model_path="bert-base")
    texts = [
        "臺灣最近在產業的部份",
        "玉山金控第一季的法人説明會"
    ]
    results = detector.predict(texts)
    print(results)