import os
from pathlib import Path
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline
from trl import SFTTrainer
import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM

HOME = "/home/koality/koarami_kurumi"

def create_directory(path: Optional[Path] = None, dir_name: str = "output"):
    """
    Creates a directory at the specified path with the given directory name.
    If no path is provided, the current working directory is used.

    Parameters:
    - path (Optional[Path]): The path where the directory is to be created.
    - dir_name (str): The name of the directory to create.

    Returns:
    - Path object representing the path to the created directory.
    """
    # Use the current working directory if no path is provided
    working_dir = path if path is not None else Path('./')

    # Define the output directory path by joining paths
    output_directory = working_dir / dir_name

    # Create the directory if it doesn't exist
    output_directory.mkdir(parents=True, exist_ok=True)

    return output_directory

def train():
    output_dir = create_directory(dir_name="fine-tuned-checkpoints")
    print(f"Created directory: {output_dir}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model_name = "Deci/DeciLM-7B-instruct"

    decilm = AutoModelForCausalLM.from_pretrained(
        model_name, 
        quantization_config=bnb_config,
        device_map="auto",
        use_cache=True,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    dataset = f"{HOME}/data/llm_datasets/train-all-4k.jsonl"
    data = load_dataset("json", data_files=dataset, split="train")
    data = data.shuffle(seed=69)
    data_split = data.train_test_split(test_size=0.1, seed=69)

    # Set lora config to be the same as qlora 
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        # The modules to apply the LoRA update matrices.
        target_modules = ["gate_proj", "down_proj", "up_proj"],
        task_type="CAUSAL_LM",
    )

    decilm = prepare_model_for_kbit_training(decilm)
    decilm = get_peft_model(decilm, lora_config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        do_eval=True,
        auto_find_batch_size=True,
        log_level="debug",
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_steps=100,
        learning_rate=3e-4,
        weight_decay=0.01,
        # Training Epochs, try 5 for now 
        max_steps=len(data_split["train"]) * 5,
        warmup_steps=150,
        bf16=True,
        tf32=True,
        gradient_checkpointing=0.3, # from paper
        lr_scheduler_type="reduce_lr_on_plateau",
    )

    trainer = SFTTrainer(
        model=decilm,
        args=training_args,
        peft_config=lora_config,
        tokenizer=tokenizer,
        dataset_text_field="text",
        train_dataset=data_split["train"],
        eval_dataset=data_split["test"],
        max_seq_length=4096,
        dataset_num_proc=os.cpu_count(),
    )

    trainer.train()
    trainer.save_model()

def merge_adapater_to_base_model():
    output_dir = create_directory(dir_name="fine-tuned-checkpoints")
    fine_tuned_model = AutoModelForCausalLM.from_pretrained(
        output_dir,
        torch_dtype=torch.bfloat16,     
        device_map="auto",
        trust_remote_code=True,
    )

    merged_model = fine_tuned_model.merge_and_unload()

    generation_kwargs = {
        "max_new_tokens": 100,
        "early_stopping": True,
        "num_beams": 5,
        "temperature": 0.001,
        "do_sample": True,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.5,
    }

    decilm_fine_tuned = pipeline(
        "text-generation",
        model=merged_model,
        tokenizer=AutoTokenizer("Deci/DeciLM-7B-instruct"),
        **generation_kwargs,
    )
    
def load_data():
    data_set = f"{HOME}/data/llm_datasets/train-all-4k.jsonl"
    data = load_dataset("json", data_files=data_set, split="train")
    data_split = data.train_test_split(test_size=0.1, seed=69)
    return data_split
    
def explore_data():
    data_set = f"{HOME}/data/llm_datasets/train-all-4k-tokenized.jsonl"
    data = load_dataset("json", data_files=data_set, split="train")
    train_dataset, test_dataset = data.train_test_split(test_size=0.1, seed=69)
    print(data.column_names)
    import pandas as pd
    df = pd.DataFrame(data_set)
    print(df.head())
    # print(data["text"][0])

if __name__ == "__main__":
    train()
    # explore_data()
    # data = load_data()
    # print(data["train"].column_names)
    # print(data["test"].column_names)
    # print(data["train"][0])