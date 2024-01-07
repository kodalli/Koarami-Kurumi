# Speech to Text

from abc import ABC, abstractmethod

class SpeechToText(ABC):

    @abstractmethod
    def hear(self, audio) -> str:
        pass