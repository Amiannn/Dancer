from typing import List

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

from pypinyin import Style
from pypinyin import pinyin
from pypinyin import lazy_pinyin

from src.utils import read_json
from src.retrieval.abs_retriever import AbsRetriever

class PinyinRetriever(AbsRetriever):
    def __init__(self, entity_path):
        self.contexts = self._load_entity(entity_path)

    def _load_entity(self, entity_path):
        contexts         = []
        entity_raw_datas = read_json(entity_path)
        if isinstance(entity_raw_datas, dict):
            for _type in entity_raw_datas:
                contexts.extend([[e, self.encode(e)] for e in entity_raw_datas[_type]])
        else:
            contexts = [[e, self.encode(e)] for e in entity_raw_datas]
        return contexts

    @classmethod
    def encode(cls, word):
        return " ".join(lazy_pinyin(word))

    @classmethod
    def similarity(cls, query, value):
        score = fuzz.ratio(query, value) / 100
        return score

    def retrieve_one_step(self, text: str, topk: int=10) -> List[str]:
        query  = self.encode(text)
        result = [] 
        for i in range(len(self.contexts)):
            key, value = self.contexts[i]
            score = self.similarity(query, value)
            result.append([score, key])
        return sorted(result, reverse=True)[:topk]
        
    def retrieve(self, texts: List[str], topk: int=10) -> List[str]:
        results = []
        for text in texts:
            result = self.retrieve_one_step(text, topk)
            results.append(result)
        return results