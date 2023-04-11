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
        align_nbest = [self.aligment(text, best) for best in nbest]
        
        nbest_prediction = []
        for align_target, align_text in align_nbest:
            align_prediction = self.shift(align_target, prediction, self.add_token)
            align_prediction = [[
                align_text[pos[0]:pos[1]].replace(self.add_token, " "), entity_type, pos
            ] for entity, entity_type, pos in align_prediction]
            nbest_prediction.append(align_prediction)
        return nbest_prediction

if __name__ == "__main__":
    target = "AXI比較特別的是它可以用X ray的原理"
    nbest  = [
        "AXI比較特別的是以用X_y ray的原理",
        "AXI比較特別的是以用Xy ray的原理",
        "AXI比較特別的的是以用X_y ray的原理"
    ]
    prediction = [["AXI", "ORG", [0, 3]], ["X ray", "ORG", [13, 18]]]

    detector = NbestDetector(model='test')
    result = detector.predict_one_step_no_detect(target, nbest, prediction)
