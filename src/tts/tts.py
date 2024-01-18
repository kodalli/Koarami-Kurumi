# Text to Speech

from abc import ABC, abstractmethod
from typing import Iterable

class TextToSpeech(ABC):

    # 16-bit mono
    @abstractmethod
    def say(self, text, **kwargs) -> Iterable[bytes]:
        pass 

    @abstractmethod
    def sample_rate(self) -> int:
        pass
