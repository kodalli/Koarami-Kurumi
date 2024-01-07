from llm import LanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch


class DeciLM7b(LanguageModel):
    def __init__(self, quantize=False):
        super().__init__()
        self.model_name = "Deci/DeciLM-7B-instruct"  # "Deci/DeciLM-7B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if quantize:
            # dtype_kwargs = dict(
            #     quantization_config=BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_compute_dtype=torch.bfloat16
            # ))
            dtype_kwargs = dict(
                quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ))
        else:
            dtype_kwargs = dict(torch_dtype="auto")

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True, **dtype_kwargs).to(
            self.device)

    def run(self, prompt, max_new_tokens=100):
        system_prompt = "You are an AI assistant that follows instruction extremely well. Help as much as you can."

        prompt = self.tokenizer.apply_chat_template([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ], tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer.encode(prompt).to(self.device)
        outputs = self.model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=True, top_p=0.95,
                                      temperature=0.7)
        return self.tokenizer.decode(outputs[0])
    
    def generate(self, prompt, max_new_tokens=100):
        pass
    
if __name__ == "__main__":
    deciLM = DeciLM7b(quantize=True) 
    response = deciLM.run(prompt="Yeehaw!")
    print(response)
    