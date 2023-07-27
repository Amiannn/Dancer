import numpy as np

from typing import List

from pypinyin import lazy_pinyin

from src.utils import read_file
from src.utils import read_json
from src.utils import write_file

from src.retrieval.abs_retriever      import AbsRetriever
from src.retrieval.pinyin_retriever   import PinyinRetriever
from src.retrieval.semantic_retriever import SemanticRetriever

class PRSRRetriever(AbsRetriever):
    def __init__(self, model_path, entity_path, entity_content_path, entity_vectors_path, alpha=0.9):
        self.semantic_retriever = SemanticRetriever(
            model_path, 
            entity_path, 
            entity_content_path, 
            entity_vectors_path
        )
        self.phonetic_retriever = PinyinRetriever(entity_path)
        
        self.overlap_contexts = []
        for entity, _ in self.phonetic_retriever.contexts:
            if entity in self.semantic_retriever.contexts:
                self.overlap_contexts.append(entity)
        
        self.phonetic_retriever.contexts = self._load_entity(self.overlap_contexts)
        self.context2idx = {entity: idx for idx, entity in enumerate(self.semantic_retriever.contexts)}

        # hyperparameter alpha
        self.alpha = alpha

    def _load_entity(self, contexts):
        contexts = [[e, self.phonetic_retriever.encode(e)] for e in contexts]
        return contexts

    @classmethod
    def encode(cls, word):
        return " ".join(lazy_pinyin(word))

    @classmethod
    def similarity(cls, query, value):
        score = fuzz.ratio(query, value) / 100
        return score

    def retrieve_one_step(self, text: str, span: List[str], topk: int=10) -> List[str]:
        phonetic_result = self.phonetic_retriever.retrieve_one_step(text, span, topk)
        
        idxs = [self.context2idx[entity] for score, entity in phonetic_result]
        self.semantic_retriever.contexts   = [entity for score, entity in phonetic_result]
        self.semantic_retriever.rank_index = self.semantic_retriever._build_rank_index(idxs)

        semantic_result = self.semantic_retriever.retrieve_one_step(text, span, topk)
        
        p_scores = {entity: score for score, entity in phonetic_result}
        s_scores = {entity: score for score, entity in semantic_result}

        result = []
        for entity in p_scores:
            ps = p_scores[entity]
            ss = s_scores[entity]
            # print(f'entity: {entity}, pinyin: {self.encode(entity)}, p_score:{ps:.3f}, s_score:{ss:.3f}')
            score = self.alpha * ps + (1 - self.alpha) * ss
            result.append([score, entity])
        # print('*' * 30)
        # for score, entity in sorted(result, reverse=True):
        #     print(f'entity: {entity}, score: {score}')
        # print('_' * 30)
        return sorted(result, reverse=True)[:topk]
        
    def retrieve(self, texts: List[str], spans: List[str], topk: int=10) -> List[str]:
        results = []
        for text, span in zip(texts, spans):
            result = self.retrieve_one_step(text, span, topk)
            results.append(result)
        return results

if __name__ == '__main__':
    model_path  = "./ckpts/ranker/dpr_biencoder.39"
    entity_path = "./datas/entities/aishell/test_1_entities.txt"
    entity_content_path = "./datas/entities/aishell/descriptions/ctx.json"
    entity_vectors_path = "./datas/entities/aishell/descriptions/embeds.npy"

    retriever = PRSRRetriever(model_path, entity_path, entity_content_path, entity_vectors_path)

    text = "韩国媒体报道称而在淮确实人在日本"
    span = [["而在淮", "", [7, 10]]]

    results = retriever.retrieve([text for _ in range(len(span))], span)
    print('_' * 30)
    print(f'original: {span[0][0]}, pinyin: {retriever.encode(span[0][0])}')
    for score, ent in results[0]:
        print(f"entity: {ent}, score: {score:.3f}")

    # path = './test_1_entities_overlap.txt'
    # entities = [[entity] for entity in retriever.overlap_contexts]
    # write_file(path, entities)