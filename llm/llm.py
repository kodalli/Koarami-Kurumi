# Language Model
from abc import ABC, abstractmethod
from typing import Iterable
import torch

class LanguageModel(ABC):
    def __init__(self):
        self.device = "gpu" if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def think(self, prompt, max_new_tokens) -> str:
        """
        Takes in a text prompt and returns a complete text response
        :param prompt: String text prompt
        :param max_new_tokens: Max tokens for LLM to return
        :return: Complete text response
        """
        pass
    
    @abstractmethod
    def generate(self, prompt) -> Iterable[str]:
        """
        Takes in a text prompt and returns a generator to stream out the text response 
        :param prompt: String text prompt
        :param max_new_tokens: Max tokens for LLM to return
        :return: Generator to stream the text response 
        """
        pass