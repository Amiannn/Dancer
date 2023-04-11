import numpy as np

from Bio    import pairwise2
from typing import List

from simpletransformers.ner import NERArgs
from simpletransformers.ner import NERModel

from src.detection.bert_detector import BertDetector

class NbestDetector(BertDetector):
    def __init__(self, model_path=None, model=None):
        if model == None:
            super().__init__(model_path)
        else:
            self.model = model
        self.add_token = "-"

    def _normalize(self, text):
        return text.replace('-', '')

    @classmethod
    def position_to_flatten(cls, text, prediction):
        flatten = [0 for t in text]
        for i in range(len(prediction)):
            ent, t, position = prediction[i]
            start, end = position
            flatten[start:end] = [i + 1 for j in range(end - start)]
        return [ent, t, flatten] 

    @classmethod
    def flatten_to_position(cls, text, datas):
        entity, t, flatten = datas
        flatten = np.array(flatten)
        length  = np.max(flatten)

        prediction = []
        for i in range(1, length + 1):
            position = np.where(flatten == i)[0]
            start, end = position[0], position[-1] + 1
            entity = text[start:end]
            prediction.append([entity, t, [start, end]])
        return prediction

    @classmethod
    def aligment(cls, target, text):
        alignments = pairwise2.align.globalxx(target, text)[0]
        # return [cls._normalize(cls, alignments.seqA), cls._normalize(cls, alignments.seqB)]
        return [alignments.seqA, alignments.seqB]

    def shift(self, text, prediction, add_token="-"):
        entity, entity_type, flatten = self.position_to_flatten(text, prediction)
        shift, flatten_shifted = 0, []
        for i in range(len(text)):
            if text[i] == add_token:
                shift += 1
            flatten_shifted.append(flatten[i - shift])
        datas = [entity, entity_type, flatten_shifted]
        prediction = self.flatten_to_position(text, datas)
        prediction = [[
            entity.replace(add_token, " "), entity_type, position
        ] for entity, entity_type, position in prediction]

        return prediction

    def predict_one_step_no_detect(self, text, nbest, prediction):
        nbest_prediction = []
        align_nbest      = [self.aligment(text, best) for best in nbest]
        for align_target, align_text in align_nbest:
            align_prediction = self.shift(align_target, prediction, self.add_token)
            align_prediction = [[
                align_text[pos[0]:pos[1]].replace(self.add_token, " "), entity_type, pos
            ] for entity, entity_type, pos in align_prediction]
            nbest_prediction.append(align_prediction)
        
        # copy prediction
        prediction_nbest = [[
            entity, entity_type, position
        ] for entity, entity_type, position in prediction]
        
        for i in range(len(nbest_prediction[0])):
            prediction_nbest[i][0] = []
            for j in range(len(nbest_prediction)):
                prediction_nbest[i][0].append(nbest_prediction[j][i][0])
        return prediction_nbest

    def predict_no_detect(self, texts, nbests, predictions):
        nbest_predictions = []
        for text, nbest, prediction in zip(texts, nbests, predictions):
            nbest_prediction = []
            if len(prediction) > 0:
                nbest_prediction = self.predict_one_step_no_detect(text, nbest, prediction)
            nbest_predictions.append(nbest_prediction)
        return nbest_predictions

    def predict_one_step(self, text: str, nbest: List[str]) -> List[str]:
        predictions = super().predict([text])[0]
        return self.predict_one_step_no_detect(text, nbest, predictions)

    def predict(self, texts: List[str], nbests: List[str]) -> List[str]:
        predictions = super().predict(texts)
        return self.predict_no_detect(texts, nbests, predictions)

if __name__ == "__main__":
    targets = [
        "AXI比較特別的是它可以用X ray的原理", 
        "阮经天和许玮甯交往八年屡传婚讯"
    ]

    nbests  = [
        [
            "AXI比較特別的是以用X_y ray的原理",
            "AXI比較特別的是以用Xy ray的原理",
            "AXI比較特別的的是以用X_y ray的原理"
        ],
        [
            "阮经天和许玮甯交往八年屡传婚讯",
            "亂今天嗎和许玮甯交往八年屡传婚讯",
            "阮经天和玮甯交往八年屡传婚讯"
        ],
    ]
    predictions = [
        [["AXI", "ORG", [0, 3]], ["X ray", "ORG", [13, 18]]],
        [["阮经天", "PRE", [0, 3]], ["X ray", "ORG", [4, 7]]]
    ]

    model_path = "/share/nas165/amian/experiments/speech/AISHELL-NER/outputs/best_model"
    detector = NbestDetector(model_path=model_path)
    result = detector.predict_one_step(targets[1], nbests[1])
    print(result)
