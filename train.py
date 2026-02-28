import os
import json
import math
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=128):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        tokenized = tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=False,
        )
        input_ids = tokenized["input_ids"][0]

        self.examples = []
        for i in range(0, len(input_ids) - block_size + 1, block_size):
            chunk = input_ids[i : i + block_size]
            self.examples.append(chunk)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {"input_ids": self.examples[idx], "labels": self.examples[idx].clone()}


def fine_tune(
    model_name="distilgpt2",
    train_file=None,
    val_file=None,
    output_dir=None,
    num_epochs=3,
    batch_size=2,
    learning_rate=2e-5,
    block_size=64,
):
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if train_file is None:
        train_file = os.path.join(PROJECT_ROOT, "data", "splits", "train.txt")
    if val_file is None:
        val_file = os.path.join(PROJECT_ROOT, "data", "splits", "val.txt")
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "models", "fine_tuned")

    print(f"Loading tokenizer and base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    print("Building datasets...")
    train_dataset = TextDataset(train_file, tokenizer, block_size)
    val_dataset = TextDataset(val_file, tokenizer, block_size)
    print(f"  Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    use_eval = len(val_dataset) > 0
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch" if use_eval else "no",
        save_strategy="epoch",
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=5,
        load_best_model_at_end=use_eval,
        metric_for_best_model="eval_loss" if use_eval else None,
        no_cuda=not torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if use_eval else None,
        data_collator=data_collator,
    )

    print(f"\nStarting fine-tuning for {num_epochs} epoch(s)...")
    train_result = trainer.train()

    print("\nSaving model and tokenizer...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    loss_log = []
    for entry in trainer.state.log_history:
        if "loss" in entry:
            loss_log.append({"step": entry.get("step"), "loss": entry["loss"]})
        elif "eval_loss" in entry:
            loss_log.append({"step": entry.get("step"), "eval_loss": entry["eval_loss"]})

    log_path = os.path.join(output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(loss_log, f, indent=2)

    print(f"\nDone! Model saved to: {output_dir}")
    print(f"Training log saved to: {log_path}")
    return train_result, loss_log


if __name__ == "__main__":
    fine_tune(num_epochs=3, batch_size=2, block_size=64)
