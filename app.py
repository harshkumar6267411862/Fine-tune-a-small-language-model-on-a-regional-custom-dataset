import os
import sys
import json
import torch
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINETUNED_DIR = os.path.join(PROJECT_ROOT, "models", "fine_tuned")
BASE_MODEL_NAME = "distilgpt2"

st.set_page_config(
    page_title="Hinglish LLM Demo",
    page_icon="🧠",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 2rem;
    color: white;
}

.main-header h1 { font-size: 2rem; margin: 0; font-weight: 700; }
.main-header p  { font-size: 0.95rem; margin: 0.5rem 0 0; opacity: 0.85; }

.output-box {
    background: #1e1e2e;
    color: #cdd6f4;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    font-size: 1rem;
    line-height: 1.7;
    border-left: 4px solid #89b4fa;
    margin-top: 1rem;
    white-space: pre-wrap;
}

.metric-card {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    color: white;
    font-weight: 600;
}

.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 2rem;
    font-size: 1rem;
    font-weight: 600;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🧠 Hinglish LLM — Fine-Tuned Demo</h1>
    <p>Distilgpt2 fine-tuned on a custom Hinglish corpus | LPU Project</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model...")
def load_model(model_path):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(model_path)
    mdl.eval()
    return tok, mdl


def generate(model, tokenizer, prompt, max_tokens, temperature, top_p):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


model_ready = os.path.exists(FINETUNED_DIR) and os.listdir(FINETUNED_DIR)

tab1, tab2, tab3 = st.tabs(["💬 Generate Text", "📊 Evaluation Results", "ℹ️ About Project"])

with tab1:
    model_choice = st.radio(
        "Choose model:",
        ["Fine-Tuned Model", "Base Model (distilgpt2)"],
        horizontal=True,
    )

    if model_choice == "Fine-Tuned Model" and not model_ready:
        st.warning("⚠️ Fine-tuned model not found. Run `python src/train.py` first, then restart the app.", icon="⚠️")
        st.stop()

    model_path = FINETUNED_DIR if model_choice == "Fine-Tuned Model" else BASE_MODEL_NAME
    tokenizer, model = load_model(model_path)

    prompt = st.text_area(
        "Enter your Hinglish prompt:",
        value="Aaj ka din bahut",
        height=80,
        max_chars=300,
    )

    with st.expander("⚙️ Generation Settings"):
        col1, col2, col3 = st.columns(3)
        with col1:
            max_tokens = st.slider("Max new tokens", 20, 200, 80)
        with col2:
            temperature = st.slider("Temperature", 0.1, 1.5, 0.8, step=0.05)
        with col3:
            top_p = st.slider("Top-p", 0.5, 1.0, 0.92, step=0.01)

    if st.button("✨ Generate"):
        if not prompt.strip():
            st.error("Please enter a prompt.")
        else:
            with st.spinner("Generating..."):
                result = generate(model, tokenizer, prompt, max_tokens, temperature, top_p)
            st.markdown(f'<div class="output-box">{result}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Try these prompts:**")
    examples = [
        "Yaar, tu kahan tha",
        "Bhai, kya tujhe pata hai",
        "Main kal college mein",
        "Aaj office mein ek badi",
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        with cols[i % 2]:
            st.code(ex)

with tab2:
    results_path = os.path.join(PROJECT_ROOT, "models", "evaluation_results.json")
    plot_path = os.path.join(PROJECT_ROOT, "models", "loss_plot.png")

    if os.path.exists(results_path):
        with open(results_path) as f:
            res = json.load(f)
        c1, c2, c3 = st.columns(3)
        c1.metric("Base Perplexity", res.get("base_perplexity", "N/A"))
        c2.metric("Fine-Tuned Perplexity", res.get("finetuned_perplexity", "N/A"))
        c3.metric("Improvement", f"{res.get('perplexity_improvement', 'N/A')}")
    else:
        st.info("Run `python src/evaluate.py` to generate evaluation results.")

    if os.path.exists(plot_path):
        st.image(plot_path, caption="Training & Validation Loss Curve", use_column_width=True)
    else:
        st.info("Training loss plot will appear here after running `python src/evaluate.py`.")

with tab3:
    st.markdown("""
### 📚 Project Overview

| Item | Detail |
|---|---|
| **University** | Lovely Professional University |
| **Course** | Python & Full Stack Development |
| **Model** | distilgpt2 (82M parameters) |
| **Dataset** | Custom Hinglish Corpus |
| **Task** | Causal Language Modelling |
| **Framework** | HuggingFace Transformers + PyTorch |

### 🚀 How to Run Full Pipeline

```bash
# Step 1 — Preprocess data
python src/dataset.py

# Step 2 — Fine-tune model
python src/train.py

# Step 3 — Evaluate & compare
python src/evaluate.py

# Step 4 — Launch this app
streamlit run app/app.py
```

### 📁 Project Structure

```
llm_fine_tune/
├── data/           raw corpus + splits
├── notebooks/      step-by-step Jupyter notebooks
├── src/            dataset · train · evaluate · inference
├── models/         saved fine-tuned model
└── app/            this Streamlit app
```
""")
