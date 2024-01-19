
from abc import ABC, abstractmethod

class VoiceToVoice(ABC):

    @abstractmethod
    def convert(self, audio, **kwargs):
        pass
    
    @abstractmethod
    def convert_file(self, audio_file, **kwargs):
        pass