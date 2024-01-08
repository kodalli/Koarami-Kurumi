from llm import LanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch


class DeciLM7b(LanguageModel):
    def __init__(self, quantize=False):
        super().__init__()
        self.model_name = "Deci/DeciLM-7B-instruct"  # "Deci/DeciLM-7B"

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
                device_map="auto",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ))
        else:
            dtype_kwargs = dict(torch_dtype="auto")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True, **dtype_kwargs)
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, temperature=0.7, top_p=0.95, max_length=4096, do_sample=True, return_full_text=False)

    def think(self, prompt):
        system_prompt = "You are an AI assistant that follows instruction extremely well. Help as much as you can."

        prompt = self.tokenizer.apply_chat_template([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ], tokenize=False, add_generation_prompt=True)

        response = self.generator(prompt)[0]["generated_text"]
        return response

        # inputs = torch.tensor(self.tokenizer.encode(prompt))
        # outputs = self.model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=True, top_p=0.95,
        #                               temperature=0.7)
        # return self.tokenizer.decode(outputs[0])
    
    def generate(self, prompt):
        pass
    
if __name__ == "__main__":
    deciLM = DeciLM7b(quantize=True) 
    text = "Yeehaw"
    response = deciLM.think(prompt=text)
    print(response)
    