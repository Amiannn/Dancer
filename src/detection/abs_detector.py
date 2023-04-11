from abc import ABC
from abc import abstractmethod

from typing import List

class AbsDetector:
    @abstractmethod
    def predict(self, texts: List[str]) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def predict_one_step(self, text: str) -> List[str]:
        raise NotImplementedError