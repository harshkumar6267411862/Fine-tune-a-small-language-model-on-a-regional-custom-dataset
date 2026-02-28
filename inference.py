import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()
    return tokenizer, model


def generate_text(model, tokenizer, prompt, device, max_new_tokens=100, temperature=0.8, top_p=0.92):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    finetuned_dir = os.path.join(PROJECT_ROOT, "models", "fine_tuned")

    if not os.path.exists(finetuned_dir) or not os.listdir(finetuned_dir):
        print("Fine-tuned model not found. Please run `python src/train.py` first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading fine-tuned model from: {finetuned_dir}")
    tokenizer, model = load_model(finetuned_dir, device)
    print("Model loaded! Type your prompt below.\n")
    print("=" * 50)
    print("Hinglish Text Generator (Fine-Tuned on Custom Corpus)")
    print("Type 'quit' to exit.")
    print("=" * 50)

    while True:
        prompt = input("\nEnter prompt: ").strip()
        if prompt.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not prompt:
            continue
        print("\nGenerating...")
        output = generate_text(model, tokenizer, prompt, device)
        print(f"\n Generated: {output}\n")
        print("-" * 50)


if __name__ == "__main__":
    main()
