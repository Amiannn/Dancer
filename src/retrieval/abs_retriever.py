from abc import ABC
from abc import abstractmethod

from typing import List

class AbsRetriever:
    @abstractmethod
    def retrieve(self, texts: List[str], topk: int=10) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def retrieve_one_step(self, text: str, topk: int=10) -> List[str]:
        raise NotImplementedError