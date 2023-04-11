from typing import List

from simpletransformers.ner import NERArgs
from simpletransformers.ner import NERModel

from src.detection.abs_detector import AbsDetector

class BertDetector(AbsDetector):
    def __init__(self, model_path, use_cuda=True):
        model = NERModel(
            "bert", model_path, args=NERArgs(), use_cuda=use_cuda
        )
        self.model = model
    
    def _preprocess(self, texts):
        return [" ".join(list(text)) for text in texts]

    def _postprocess(self, datas):
        s, e, t = 0, 0, None
        entity_pos, entity_datas, text = [], [], []

        for i in range(len(datas)):
            char, _type = list(datas[i].keys())[0], list(datas[i].values())[0]
            entity_datas.append([char, _type])
            text.append(char)

        for i in range(len(entity_datas)):
            char, _type = entity_datas[i]
            e = i
            if "B-" in _type:
                s = i
                t = _type.split('-')[-1]
            elif _type == "O":
                if t != None:
                    entity_pos.append(["".join(text[s:e]), t, [s, e]])
                s = i
                t = None
        return entity_pos

    def predict(self, texts: List[str]) -> List[str]:
        texts = self._preprocess(texts)
        predictions, raw_outputs = self.model.predict(texts)
        predictions = [self._postprocess(pred) for pred in predictions]
        return predictions

    def predict_one_step(self, text: str) -> List[str]:
        return self.predict([text])[0]
