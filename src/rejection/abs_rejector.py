from abc import ABC
from abc import abstractmethod

from typing import List

class AbsRejector:
    @abstractmethod
    def reject_one_step(self, text: str) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def reject(self, candiate: str, nbest: List[str]) -> List[str]:
        raise NotImplementedError