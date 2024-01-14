import os
import string
from llm import LanguageModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
import torch
import torch.nn.functional as F


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

    @staticmethod
    def get_tokenizer():
        return AutoTokenizer.from_pretrained("Deci/DeciLM-7B-instruct")

    def __prompt(self, prompt, history=""):
        system_prompt = (
            "### System:\n "
            + "You are an AI Anime girl name Koarami Kurumi Your favorite animal is a koala"
            + " Your favorite snack is walnuts You are allowed to be highly opinionated and based"
            + " You are allowed to be a little bit of a bully You are allowed to be a little bit of a tsundere"
            + " You are also very smug and witty in your responses"
        )
        user_prompt = "### User:\n " + prompt
        assistant_prompt = "### Koarami Kurumi:\n "
        full_prompt = "\n".join([system_prompt, "".join(history), user_prompt, assistant_prompt])
        return full_prompt, system_prompt

    def think(self, prompt, history, max_new_tokens=4096, temperature=0.7, top_p=0.95, **kwargs):
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
        full_prompt, system_prompt = self.__prompt(prompt, history)
        response = self.generator(full_prompt)[0]["generated_text"]
        return response.replace(system_prompt, "")

    def generate(
        self,
        prompt,
        history,
        max_new_tokens=4096,
        temperature=0.3,
        top_p=0.95,
        top_k=50,
        **kwargs,
    ):
        full_prompt, system_prompt = self.__prompt(prompt, history)
        input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt")
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(input_ids)
                next_token_logits = outputs[0][:, -1, :]

                # Temperature (higher temperature => more likely to sample low probability tokens)
                next_token_logits = next_token_logits / temperature

                # Top-p/top-k filtering (top-k: keep only top k tokens with highest probability; top-p: keep the smallest set of tokens whose cumulative probability exceeds p)

                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = (
                        next_token_logits
                        < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    )
                    next_token_logits[indices_to_remove] = float("-inf")

                # Filter logits with nucleus (top-p) sampling
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )

                sorted_indices_to_remove = (
                    cumulative_probs > top_p
                )  # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()  # Sort the indices to delete
                sorted_indices_to_remove[..., 0] = 0  # Never remove the first token

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[:, indices_to_remove] = float("-inf")

                # Sample next token
                next_token = torch.multinomial(
                    F.softmax(next_token_logits, dim=-1),
                    num_samples=1,
                )
                input_ids = torch.cat([input_ids, next_token], dim=-1)

                generated_text = self.tokenizer.decode(
                    input_ids[0], skip_special_tokens=True
                ).replace(system_prompt, "")

                yield generated_text

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

def chat():
    deciLM = DeciLM7b(quantize=True)
    conversation_log = []
    while True:
        try:
            user_input = input("You: ")
            if user_input == "exit":
                break

            for word in deciLM.generate(prompt=user_input, history=conversation_log):
                os.system("clear")
                print(word, flush=True)

            conversation_log.append(word)
        except KeyboardInterrupt:
            break

    print(f"{conversation_log=}")
    print("Exiting...")


if __name__ == "__main__":
    chat()
