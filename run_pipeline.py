"""
run_pipeline.py — Full pipeline runner (preprocesses data, fine-tunes, evaluates)
Run this single script to do everything end-to-end.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))


def step(title):
    print("\n" + "=" * 55)
    print(f"  {title}")
    print("=" * 55)


if __name__ == "__main__":
    from dataset import preprocess_pipeline

    step("STEP 1/3 — Data Preprocessing")
    train_path, val_path = preprocess_pipeline(
        raw_path=os.path.join(PROJECT_ROOT, "data", "raw", "hinglish_corpus.txt"),
        cleaned_dir=os.path.join(PROJECT_ROOT, "data", "cleaned"),
        splits_dir=os.path.join(PROJECT_ROOT, "data", "splits"),
    )

    step("STEP 2/3 — Fine-Tuning distilgpt2 (laptop-optimised)")
    from train import fine_tune
    fine_tune(
        model_name="distilgpt2",
        train_file=train_path,
        val_file=val_path,
        num_epochs=3,
        batch_size=2,       # small batch for laptop RAM
        learning_rate=2e-5,
        block_size=64,      # shorter context for speed
    )

    step("STEP 3/3 — Evaluation (base vs fine-tuned)")
    from evaluate import evaluate
    results = evaluate(base_model_name="distilgpt2")

    print("\n" + "=" * 55)
    print("  ALL DONE!")
    print(f"  Base Perplexity     : {results['base_perplexity']}")
    print(f"  Fine-Tuned Perplexity: {results['finetuned_perplexity']}")
    print(f"  Improvement          : {results['perplexity_improvement']}")
    print("=" * 55)
    print("\nNext step: streamlit run app/app.py")
