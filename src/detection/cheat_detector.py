import numpy as np

from Bio    import Align
from typing import List

from simpletransformers.ner import NERArgs
from simpletransformers.ner import NERModel

from src.utils import read_file
from src.utils import read_json
from src.detection.abs_detector import AbsDetector

class CheatDetector(AbsDetector):
    def __init__(self, entity_path):
        self.entities = self._load_entity(entity_path)
        self.add_token = "-"
        self.aligner = self._get_aligner()

    def _load_entity(self, entity_path):
        contexts = []
        contexts = read_file(entity_path)
        contexts = [e[0] for e in contexts]
        return list(sorted(contexts, reverse=True))
    
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

    def _get_aligner(self):
        aligner = Align.PairwiseAligner()
        aligner.match_score    = 1.0 
        aligner.gap_score      = -2.5
        aligner.mismatch_score = -2.0
        return aligner

    def aligment(self, target, text):
        alignments = self.aligner.align(target, text)[0]
        alignments = str(alignments).split('\n')
        seqA, seqB = alignments[0], alignments[2]
        return [seqA, seqB]

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

    def find_all_place(self, text, subtext):
        data = []
        now = text.find(subtext)
        while(now >= 0):
            data.append([now, now + len(subtext)])
            now = text.find(subtext, now + 1)
        return data

    def check_position_hited(self, hitmap, position):
        start, end = position
        delta_hitmap = np.array([0 for t in range(hitmap.shape[0])])
        delta_hitmap[start:end] = 1

        if np.sum(hitmap * delta_hitmap) > 0:
            return True, delta_hitmap
        return False, delta_hitmap

    def find_entity_mention(self, text):
        datas  = []
        hitmap = np.array([0 for t in text])
        for entity in self.entities:
            positions = self.find_all_place(text, entity)
            for position in positions:
                ifpass, delta_hitmap = self.check_position_hited(hitmap, position)
                if ifpass:
                    continue
                else:
                    hitmap += delta_hitmap
                    datas.append([entity, 'CHEAT', position])
        # sort it
        datas = sorted([[data[-1][0], data] for data in datas])
        datas = [data[1] for data in datas]
        return datas

    def predict_one_step(self, target: str, text: str) -> List[str]:
        target_prediction        = self.find_entity_mention(target)
        align_target, align_text = self.aligment(target, text)
        
        if len(target_prediction) > 0:
            align_prediction  = self.shift(align_target, target_prediction, self.add_token)
            align_prediction  = [[
                align_text[pos[0]:pos[1]].replace(self.add_token, " "), entity_type, pos
            ] for entity, entity_type, pos in align_prediction]
        else:
            align_prediction = target_prediction
        return align_prediction

    def predict(self, targets: List[str], texts: List[str]) -> List[str]:
        predictions = [self.predict_one_step(target, text) for target, text in zip(targets, texts)]
        return predictions

if __name__ == "__main__":
    entity_path = "/share/nas165/amian/experiments/speech/EntityCorrector/blists/aishell/test_1_entities.txt"
    detector = CheatDetector(entity_path)

    target = "每日经济新闻记者杨建江南嘉捷六万"
    text   = "每日经济新闻记者杨建江南嘉捷六万"

    prediction = detector.predict_one_step(target, text)
    print(prediction)