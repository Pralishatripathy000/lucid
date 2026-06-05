import torch
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

HF_TOKEN = "your_hf_token_here"

wandb.init(project="lucid-hallucination-detector", name="lucid-layer2-llama32")

print("Loading dataset...")
train_data = load_dataset("json", data_files="./data/processed/train.jsonl", split="train")
val_data   = load_dataset("json", data_files="./data/processed/val.jsonl", split="train")
print(f"Train: {len(train_data)}, Val: {len(val_data)}")

def format_prompt(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    ),
    device_map="auto",
    trust_remote_code=True,
    token=HF_TOKEN
)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model.config.use_cache = False
print("Model loaded!")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B", token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    formatting_func=format_prompt,
    peft_config=LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj","v_proj","k_proj","o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    ),
    args=TrainingArguments(
        output_dir="./models/lucid-layer2",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        bf16=True,
        fp16=False,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="wandb",
        run_name="lucid-layer2-llama32"
    )
)

print("Starting training...")
trainer.train()
trainer.save_model("./models/lucid-layer2")
print("Training complete!")
wandb.finish()