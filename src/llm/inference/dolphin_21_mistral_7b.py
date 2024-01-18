
from typing import Iterable
from src.llm.inference.llm import LanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch


class DolphinMistral7b(LanguageModel):
    def __init__(self, quantize=True):
        super().__init__()
        self.model_name = "cognitivecomputations/dolphin-2.1-mistral-7b"

        if quantize:
            dtype_kwargs = dict(
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    device_map="auto",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            )
        else:
            dtype_kwargs = dict(torch_dtype="auto")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, trust_remote_code=True, **dtype_kwargs
        )

    def __prompt(self, prompt, history=""):
        system_prompt = "\n".join([ 
           "<|im_start|>system",
            "You are Dolphin, a helpful AI assistant.<|im_end|>",
            "<|im_start|>user",
            f"{prompt}<|im_end|>",
            "<|im_start|>assistant"
        ])
        return system_prompt

    def think(self, prompt, history="", max_new_tokens=4096, temperature=0.7, top_p=0.95, **kwargs):
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            temperature=temperature,
            top_p=top_p,
            max_length=max_new_tokens,
            do_sample=True,
            return_full_text=False,
        )
        full_prompt = self.__prompt(prompt, history)
        response = self.generator(full_prompt)[0]["generated_text"]
        return response

    def generate(self, prompt, **kwargs) -> Iterable[str]:
        pass