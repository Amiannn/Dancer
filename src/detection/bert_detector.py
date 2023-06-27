import numpy as np

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
    
    def _get_scores(self, raw_predicts, predicts):
        scores = []
        for raw_predict, predict in zip(raw_predicts, predicts):
            cl_scores = []
            for raw in raw_predict:
                word  = list(raw.keys())[0]
                logit = np.array(list(raw.values())[0][0], dtype=np.float32)
                prob  = np.exp(logit) / np.sum(np.exp(logit))
                # no-entity or entity
                prob  = np.array([prob[0], np.sum(prob[1:])])
                prob  = prob / np.sum(prob)
                score = prob[1]
                cl_scores.append(score)
            print()
            wl_scores = []
            for entity, _, position in predict:
                start, end = position
                wl_score = sum(cl_scores[start:end]) / (end - start)
                wl_scores.append(wl_score)
            scores.append(wl_scores)
        return scores

    def predict(self, texts: List[str]) -> List[str]:
        texts = self._preprocess(texts)
        predictions, raw_outputs = self.model.predict(texts)
        predictions = [self._postprocess(pred) for pred in predictions]
        scores = self._get_scores(raw_outputs, predictions)
        return predictions, scores

    def predict_one_step(self, text: str) -> List[str]:
        return self.predict([text])[0]

if __name__ == '__main__':
    model_path = "/share/nas165/amian/experiments/speech/AISHELL-NER/outputs/best_model"
    detector = BertDetector(model_path)

    text = ["许玮拎日前传闻阮經天八年情变", "每日经济新闻记者杨建江南佳节六万"]
    detector.predict(text)
