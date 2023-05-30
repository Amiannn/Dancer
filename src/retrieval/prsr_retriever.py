from typing import List

from src.utils import read_file
from src.utils import read_json

from src.retrieval.abs_retriever      import AbsRetriever
from src.retrieval.pinyin_retriever   import PinyinRetriever
from src.retrieval.semantic_retriever import SemanticRetriever

class PRSRRetriever(AbsRetriever):
    def __init__(self, model_path, entity_path, entity_content_path, entity_vectors_path):
        self.semantic_retriever = SemanticRetriever(
            model_path, 
            entity_path, 
            entity_content_path, 
            entity_vectors_path
        )
        self.phonetic_retriever = PinyinRetriever(entity_path)
        self.phonetic_retriever.contexts = self._load_entity(self.semantic_retriever.contexts)

        self.context2idx = {entity: idx for idx, entity in enumerate(self.semantic_retriever.contexts)}

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
        phonetic_result = self.phonetic_retriever.retrieve_one_step(text, span)
        
        idxs = [self.context2idx[entity] for score, entity in phonetic_result]
        self.semantic_retriever.contexts   = [entity for score, entity in phonetic_result]
        self.semantic_retriever.rank_index = self.semantic_retriever._build_rank_index(idxs)

        semantic_result = self.semantic_retriever.retrieve_one_step(text, span)
        
        p_scores = {entity: score for score, entity in phonetic_result}
        s_scores = {entity: score for score, entity in semantic_result}

        result = []
        alpha  = 0.999
        for entity in p_scores:
            ps = p_scores[entity]
            ss = s_scores[entity]
            score = alpha * ps + (1 - alpha) * ss
            result.append([score, entity])
        return sorted(result, reverse=True)[:topk]
        
    def retrieve(self, texts: List[str], spans: List[str], topk: int=10) -> List[str]:
        results = []
        for text, span in zip(texts, spans):
            result = self.retrieve_one_step(text, span, topk)
            results.append(result)
        return results

if __name__ == '__main__':
    model_path  = "/share/nas165/amian/experiments/nlp/DPR/outputs/2023-03-07/16-45-46/output/dpr_biencoder.39"
    entity_path = "/share/nas165/amian/experiments/speech/EntityCorrector/blists/aishell/test_1_entities.txt"
    entity_content_path = "/share/nas165/amian/experiments/speech/AISHELL-NER/dump/2023_02_27__15_58_45_test/aishell_ner_ctx.json"
    entity_vectors_path = "/share/nas165/amian/experiments/speech/AISHELL-NER/dump/2023_15_03__14_36_47/embeds.npy"

    retriever = PRSRRetriever(model_path, entity_path, entity_content_path, entity_vectors_path)

    text = "许玮拎日前传闻阮經天八年情变"
    span = [["许玮拎", "", [0, 3]], ["阮經天", "", [7, 10]]]

    results = retriever.retrieve([text for _ in range(len(span))], span)
    print(results)