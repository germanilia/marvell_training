#!/usr/bin/env python3
from dataclasses import dataclass, field
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
import os
import subprocess
import sys
from typing import Optional, List

from optimum.neuron import NeuronHfArgumentParser as HfArgumentParser
from optimum.neuron import NeuronSFTConfig, NeuronSFTTrainer, NeuronTrainingArguments
from optimum.neuron.distributed import lazy_load_for_parallelism
from torch_xla.core.xla_model import is_master_ordinal


def check_requirements():
    required_packages = {
        'transformers': 'transformers',
        'datasets': 'datasets',
        'peft': 'peft',
        'optimum.neuron': 'optimum-neuron',
        'torch_xla': 'torch-xla',
    }
    
    missing = []
    for package, pip_name in required_packages.items():
        try:
            __import__(package.split('.')[0])
        except ImportError:
            missing.append(pip_name)
    
    if missing:
        raise ImportError(
            f"Required packages {', '.join(missing)} are missing. "
            f"Please install them using: pip install {' '.join(missing)}"
        )


@dataclass
class ScriptArguments:
    model_id: str = field(
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub."
        },
    )
    tokenizer_id: str = field(
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        metadata={"help": "The tokenizer used to tokenize text for fine-tuning."},
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA r value to be used during fine-tuning."},
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha value to be used during fine-tuning."},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout value to be used during fine-tuning."},
    )


def training_function(script_args, training_args):
    # Load and prepare dataset
    dataset = load_dataset("b-mc2/sql-create-context", split="train")
    dataset = dataset.shuffle(seed=23)
    train_dataset = dataset.select(range(50000))
    eval_dataset = dataset.select(range(50000, 50500))

    def create_conversation(sample):
        system_message = (
            "You are a text to SQL query translator. Users will ask you questions in English and you will generate a "
            "SQL query based on the provided SCHEMA.\nSCHEMA:\n{schema}"
        )
        return {
            "messages": [
                {
                    "role": "system",
                    "content": system_message.format(schema=sample["context"]),
                },
                {"role": "user", "content": sample["question"]},
                {"role": "assistant", "content": sample["answer"] + ";"},
            ]
        }

    train_dataset = train_dataset.map(
        create_conversation, remove_columns=train_dataset.features, batched=False
    )
    eval_dataset = eval_dataset.map(
        create_conversation, remove_columns=eval_dataset.features, batched=False
    )

    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_id)

    with lazy_load_for_parallelism(
        tensor_parallel_size=training_args.tensor_parallel_size
    ):
        model = AutoModelForCausalLM.from_pretrained(script_args.model_id)

    config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "q_proj",
            "gate_proj",
            "v_proj",
            "o_proj",
            "k_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False,
        init_lora_weights=True
    )

    # Create base config without output_dir
    base_config = {
        "max_seq_length": 1024,
        "packing": True,
        "dataset_kwargs": {
            "add_special_tokens": False,
            "append_concat_token": True,
        }
    }

    # Remove output_dir from training args to avoid duplicate
    training_dict = training_args.to_dict()
    if 'output_dir' in training_dict:
        del training_dict['output_dir']

    # Merge configurations
    config_dict = {**base_config, **training_dict}
    
    # Create SFT config with output_dir as a separate parameter
    sft_config = NeuronSFTConfig(
        output_dir=training_args.output_dir,
        **config_dict
    )

    trainer = NeuronSFTTrainer(
        args=sft_config,
        model=model,
        peft_config=config,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()


if __name__ == "__main__":
    # Check requirements first
    check_requirements()
    
    # Parse arguments
    parser = HfArgumentParser([ScriptArguments, NeuronTrainingArguments])
    script_args, training_args = parser.parse_args_into_dataclasses()

    # Configure training settings
    training_args.report_to = []  # Disable wandb or other reporting
    training_args.disable_tqdm = True
    training_args.save_strategy = "steps"
    training_args.save_steps = 500
    training_args.save_total_limit = 2
    training_args.save_safetensors = True

    # Set random seed for reproducibility
    set_seed(training_args.seed)
    
    # Run training
    training_function(script_args, training_args)

    # Handle model merging (only on master process)
    if is_master_ordinal():
        input_ckpt_dir = os.path.join(
            training_args.output_dir, f"checkpoint-{training_args.max_steps}"
        )
        output_ckpt_dir = os.path.join(training_args.output_dir, "merged_model")
        
        subprocess.run(
            [
                "python3",
                "consolidate_adapter_shards_and_merge_model.py",
                "-i",
                input_ckpt_dir,
                "-o",
                output_ckpt_dir,
            ]
        )