from abc import ABC
from abc import abstractmethod

from typing import List

from src.rejection.abs_rejector     import AbsRejector
from src.retrieval.pinyin_retriever import PinyinRetriever

class NbestRejector(AbsRejector):
    def score_normalize(length, n, score):
        b     = sum(list(range(length)))
        ratio = (length - n) / b if b > 0 else 0
        return (1 - score) * ratio

    def reject_one_step(self, candiate: str, nbest: List[str]) -> List[str]:
        rejection_score = 0
        for i, best in enumerate(nbest):
            query = PinyinRetriever.encode(candiate)
            value = PinyinRetriever.encode(best)
            score = PinyinRetriever.similarity(query, value)
            rejection_score += self.score_normalize(len(nbest), i + 1, score)
        return rejection_score

    @classmethod
    def reject(cls, prediction: List[str], prediction_nbest: List[str], candiates: List[str]) -> List[str]:
        results = []
        for i in range(len(prediction)):
            candiate = candiates[i]
            original, t, position = prediction[i]
            nbest, _, _ = prediction_nbest[i]

            candiate_rejection_score = cls.reject_one_step(cls, candiate, nbest)
            original_rejection_score = cls.reject_one_step(cls, original, nbest)

            final = original
            if candiate_rejection_score <= original_rejection_score:
                final = candiate
            results.append(final)
        return results
