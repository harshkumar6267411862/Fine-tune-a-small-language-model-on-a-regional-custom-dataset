import os
import math
import json
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_path_or_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path_or_name).to(device)
    model.eval()
    return tokenizer, model


def compute_perplexity(model, tokenizer, text_file, device, block_size=128):
    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read()

    encodings = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = encodings["input_ids"][0].to(device)

    total_loss = 0.0
    num_chunks = 0

    with torch.no_grad():
        for i in range(0, len(input_ids) - block_size + 1, block_size):
            chunk = input_ids[i : i + block_size].unsqueeze(0)
            outputs = model(chunk, labels=chunk)
            total_loss += outputs.loss.item()
            num_chunks += 1

    avg_loss = total_loss / max(num_chunks, 1)
    perplexity = math.exp(avg_loss)
    return perplexity, avg_loss


def generate_sample(model, tokenizer, prompt, device, max_new_tokens=60):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.92,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text


def plot_training_loss(log_path, save_path):
    if not os.path.exists(log_path):
        print("No training log found, skipping loss plot.")
        return

    with open(log_path, "r") as f:
        log = json.load(f)

    train_steps = [e["step"] for e in log if "loss" in e]
    train_losses = [e["loss"] for e in log if "loss" in e]
    eval_steps = [e["step"] for e in log if "eval_loss" in e]
    eval_losses = [e["eval_loss"] for e in log if "eval_loss" in e]

    plt.figure(figsize=(10, 5))
    if train_steps:
        plt.plot(train_steps, train_losses, label="Training Loss", marker="o", color="steelblue")
    if eval_steps:
        plt.plot(eval_steps, eval_losses, label="Validation Loss", marker="s", color="tomato")
    plt.title("Training & Validation Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Loss plot saved to: {save_path}")


def evaluate(base_model_name="distilgpt2", finetuned_model_dir=None):
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if finetuned_model_dir is None:
        finetuned_model_dir = os.path.join(PROJECT_ROOT, "models", "fine_tuned")

    val_file = os.path.join(PROJECT_ROOT, "data", "splits", "val.txt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n[1/3] Loading BASE model...")
    base_tok, base_model = load_model(base_model_name, device)

    print("[2/3] Loading FINE-TUNED model...")
    ft_tok, ft_model = load_model(finetuned_model_dir, device)

    print("\n--- Perplexity Comparison ---")
    base_ppl, base_loss = compute_perplexity(base_model, base_tok, val_file, device)
    ft_ppl, ft_loss = compute_perplexity(ft_model, ft_tok, val_file, device)
    print(f"  Base model perplexity:        {base_ppl:.2f}  (loss: {base_loss:.4f})")
    print(f"  Fine-tuned model perplexity:  {ft_ppl:.2f}  (loss: {ft_loss:.4f})")

    prompts = [
        "Aaj ka din bahut",
        "Yaar, tu kahan",
        "Bhai, kya tujhe pata hai",
        "Main kal college mein",
    ]

    print("\n--- Qualitative Output Comparison ---")
    for prompt in prompts:
        print(f"\n  Prompt: \"{prompt}\"")
        base_out = generate_sample(base_model, base_tok, prompt, device)
        ft_out = generate_sample(ft_model, ft_tok, prompt, device)
        print(f"  Base     : {base_out}")
        print(f"  Fine-Tuned: {ft_out}")

    log_path = os.path.join(finetuned_model_dir, "training_log.json")
    plot_path = os.path.join(PROJECT_ROOT, "models", "loss_plot.png")
    plot_training_loss(log_path, plot_path)

    results = {
        "base_perplexity": round(base_ppl, 2),
        "finetuned_perplexity": round(ft_ppl, 2),
        "perplexity_improvement": round(base_ppl - ft_ppl, 2),
    }

    results_path = os.path.join(PROJECT_ROOT, "models", "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nEvaluation results saved to: {results_path}")
    return results


if __name__ == "__main__":
    evaluate()
