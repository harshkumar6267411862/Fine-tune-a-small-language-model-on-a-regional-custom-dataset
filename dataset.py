import os
import re
import random


def load_raw_text(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if line.strip()]
    return lines


def clean_text(lines):
    cleaned = []
    seen = set()
    for line in lines:
        line = re.sub(r"http\S+|www\S+", "", line)
        line = re.sub(r"<[^>]+>", "", line)
        line = re.sub(r"\s+", " ", line).strip()
        if line and line not in seen:
            seen.add(line)
            cleaned.append(line)
    return cleaned


def save_text(lines, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def train_val_split(lines, val_ratio=0.1, seed=42):
    random.seed(seed)
    random.shuffle(lines)
    split_idx = int(len(lines) * (1 - val_ratio))
    return lines[:split_idx], lines[split_idx:]


def preprocess_pipeline(raw_path, cleaned_dir, splits_dir):
    print(f"Loading data from: {raw_path}")
    lines = load_raw_text(raw_path)
    print(f"  Loaded {len(lines)} lines")

    print("Cleaning and deduplicating...")
    lines = clean_text(lines)
    print(f"  Retained {len(lines)} clean lines")

    cleaned_path = os.path.join(cleaned_dir, "cleaned_corpus.txt")
    save_text(lines, cleaned_path)
    print(f"  Saved cleaned data to: {cleaned_path}")

    train_lines, val_lines = train_val_split(lines)
    train_path = os.path.join(splits_dir, "train.txt")
    val_path = os.path.join(splits_dir, "val.txt")
    save_text(train_lines, train_path)
    save_text(val_lines, val_path)
    print(f"  Train: {len(train_lines)} lines -> {train_path}")
    print(f"  Val:   {len(val_lines)} lines -> {val_path}")

    return train_path, val_path


if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    preprocess_pipeline(
        raw_path=os.path.join(PROJECT_ROOT, "data", "raw", "hinglish_corpus.txt"),
        cleaned_dir=os.path.join(PROJECT_ROOT, "data", "cleaned"),
        splits_dir=os.path.join(PROJECT_ROOT, "data", "splits"),
    )
