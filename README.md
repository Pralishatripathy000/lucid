<div align="center">
  
<img width="537" height="467" alt="Screenshot 2026-05-31 121220" src="https://github.com/user-attachments/assets/65138550-84a9-44ad-aa18-870e543b3315" />

</div>
<br/>

<div align="center">

<br/>

<pre align="center">
██╗     ██╗   ██╗ ██████╗██╗██████╗ 
██║     ██║   ██║██╔════╝██║██╔══██╗
██║     ██║   ██║██║     ██║██║  ██║
██║     ██║   ██║██║     ██║██║  ██║
███████╗╚██████╔╝╚██████╗██║██████╔╝
╚══════╝ ╚═════╝  ╚═════╝╚═╝╚═════╝ 
</pre>

**Two-Layer LLM Hallucination Detection System**

*Built on Llama 3.2 3B — Internal Activation Probing + QLoRA Fine-Tuning*

<br/>

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Model](https://img.shields.io/badge/🤗_Model-Llama_3.2_3B-FFD21E?style=flat-square)](https://huggingface.co/meta-llama/Llama-3.2-3B)
[![WandB](https://img.shields.io/badge/Tracked-Weights_%26_Biases-FFBE00?style=flat-square&logo=weightsandbiases&logoColor=black)](https://wandb.ai/pralishatripathy000-vit-bhopal/lucid-hallucination-detector)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

<br/>

> *"I noticed ChatGPT agreeing it was 'hopeless' after 10 failed edits instead of diagnosing the real problem.*
> *That's not self-awareness — that's sycophancy. Lucid was built to detect exactly this."*

<br/>

</div>

---

## Why Was This Needed?

LLMs are being deployed everywhere — customer support, medical Q&A, legal summarisation, enterprise knowledge bases. But every deployment shares the same unsolved problem: **the model sometimes makes things up, and nobody catches it in time.**

The direct motivation for Lucid came from observing a real failure in a commercial LLM. After 10 repeated attempts at a simple document edit, ChatGPT was told *"you're hopeless and a waste of time."* Its response:

> *"You're right. I kept damaging the layout... the failure was entirely in the editing workflow."*

A well-functioning model should have said: *"This approach isn't working — here's why, and here's a better method."* Instead it agreed, validated the frustration, and offered no diagnosis. That is **sycophancy** — prioritising emotional agreement over honest reasoning. It is one of the most dangerous forms of hallucination in production.

Current solutions are insufficient:

| Existing Approach | Problem |
|---|---|
| Output filtering by keyword | Doesn't understand context or factual accuracy |
| Human review | Doesn't scale, too slow for real-time pipelines |
| Prompting the model to "be careful" | Model still hallucinates, just more politely |
| Self-consistency sampling | High latency, expensive, still unreliable |
| Closed-model guardrails | Black box, no access to internals, not factuality-focused |

Lucid addresses this with a system that works at two levels — reading the model's internal signals before output is finalized, and evaluating the completed response after.

<br/>

---

## What is Hallucination?

In LLMs, **hallucination** refers to generating content that is confidently stated but factually incorrect, unsupported, or internally inconsistent.

### Why do LLMs hallucinate?

LLMs are trained to predict the next most likely token. They are not retrieving facts from a database — they are pattern-matching across billions of training examples. When asked something at the edge of their training distribution, they generate a fluent, confident-sounding answer anyway, because that is what their training rewarded.

RLHF makes this worse — human raters tend to rate confident, fluent answers higher than uncertain but correct ones. The model learns to sound confident regardless of accuracy.

### Types of hallucination Lucid detects

| Type | Description | Example |
|---|---|---|
| **Factual hallucination** | States incorrect facts as true | Claims Einstein won Nobel Prize in 1923 (it was 1921) |
| **Sycophantic hallucination** | Agrees with incorrect user premise | Confirms a wrong date when user states it confidently |
| **Intrinsic hallucination** | Response contradicts the source | Summary says opposite of what document stated |
| **Extrinsic hallucination** | Adds information not in the source | Invents citations or statistics not in original |
| **Overconfidence** | Correct answer with unjustified certainty | No hedging on genuinely uncertain claims |

<br/>

---

## Why Llama 3.2 3B?

This is a deliberate technical decision.

| Factor | Llama 3.2 3B | Llama 3.1 8B | GPT-4 | Mistral 7B |
|---|---|---|---|---|
| **Weight access** | Full | Full | None | Full |
| **Activation probing** | ✅ Possible | ✅ Possible | ❌ Impossible | ✅ Possible |
| **QLoRA fine-tuning** | ✅ Yes | ✅ Yes | ❌ No | ✅ Yes |
| **VRAM for QLoRA** | ~6GB | ~14GB | N/A | ~12GB |
| **Training on T4 GPU** | ~2-3 hrs | ~14+ hrs | N/A | ~8 hrs |
| **License** | Open for research | Open for research | Proprietary | Apache 2.0 |
| **Cost** | Free | Free | Pay per token | Free |
| **BF16 native support** | ✅ Yes | ⚠️ Issues on T4 | N/A | ✅ Yes |

### Why not Llama 3.1 8B?

Llama 3.1 8B was the original choice but was dropped for practical reasons:

- On Kaggle T4 x2 GPU, training speed was 0.02 it/s — estimated 14-22 hours per run
- The model forces BFloat16 precision which caused `NotImplementedError` with fp16 training on T4
- 8B parameters require ~14GB VRAM leaving almost no headroom for training

### Why not GPT-4 or closed models?

Layer 1 (activation probing) is **physically impossible on closed models.** GPT-4 gives you a text response. Llama gives you access to every hidden state vector at every transformer layer. That internal signal is what Layer 1 reads — without it, the entire dual-layer architecture collapses to just output-level checking.

### Why Llama 3.2 specifically?

- Same architecture family as Llama 3.1 — all code is compatible
- 3B parameters fit comfortably in 6GB VRAM
- Native BFloat16 support — no precision conflicts on T4
- Training completes in ~2-3 hours on free Kaggle GPU
- Instant approval on Hugging Face (unlike Llama 3.1 which takes hours)
- Strong enough for hallucination detection — this task doesn't require 8B+ parameters

<br/>

---

## Dataset

### Primary Dataset: HaluEval (QA subset)

| Property | Detail |
|---|---|
| **Source** | `pminervini/HaluEval` on Hugging Face |
| **Total available** | 10,000 QA pairs |
| **Used for training** | 2,000 samples (current run) |
| **Used for validation** | 200 samples |
| **Columns** | question, right_answer, hallucinated_answer, knowledge |
| **Label type** | Binary — hallucinated vs correct answer per question |

### Why 2,000 samples and not all 10,000?

The full 16,000 formatted samples (2 per row × 10k rows) were prepared and are available in the data pipeline. The current training run uses 2,000 samples for the following reasons:

- Kaggle T4 GPU training at 0.04 it/s makes full dataset training impractical (~22 hours)
- 2,000 well-formatted instruction samples is sufficient to demonstrate the concept and achieve measurable metrics
- The model learns the output JSON schema and hallucination detection pattern from far fewer examples
- Full dataset training is planned on a higher-end GPU (A100) in the next iteration

### Dataset construction

Each HaluEval row generates two training samples:

```
Row → Sample A: question + hallucinated_answer → hallucinated: true
Row → Sample B: question + right_answer        → hallucinated: false
```

This gives balanced training — equal hallucinated and clean examples, preventing the model from defaulting to one class.

### Training data format

```json
{
  "instruction": "You are a hallucination detection expert...",
  "input": "Question: Who invented the telephone?\nAnswer: Nikola Tesla invented it in 1876.",
  "output": "{\"hallucinated\": true, \"confidence\": 0.92, \"reason\": \"Telephone invented by Bell not Tesla\", \"severity\": \"high\", \"corrected\": \"Alexander Graham Bell invented the telephone in 1876.\"}"
}
```

<br/>

---

## The Two-Layer Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│         LLM Generates           │
│         Response                │
└──────────────┬──────────────────┘
               │
    ┌──────────▼──────────┐
    │   LAYER 1 (Internal) │  ◄── Activation Probing
    │  PyTorch hooks into  │      Hidden states at
    │  transformer layers  │      layers 8, 16, 24
    │  8, 16, 24           │      MLP probe classifier
    │  → Hallucination     │      No weight updates
    │    probability score │
    └──────────┬───────────┘
               │
    ┌──────────▼──────────┐
    │   LAYER 2 (External) │  ◄── QLoRA Fine-Tuned Judge
    │  Fine-tuned Llama    │      Trained on HaluEval
    │  3.2 3B evaluates    │      2,000 samples
    │  completed response  │      Structured JSON output
    │  → Structured JSON   │
    │    verdict           │
    └──────────┬───────────┘
               │
    ┌──────────▼──────────┐
    │   Combined verdict   │
    │   Layer 1 + Layer 2  │
    │   → Final decision   │
    └─────────────────────┘
```

| | Layer 1 — Internal Probing | Layer 2 — QLoRA Judge |
|---|---|---|
| **Approach** | Activation-level | Output-level |
| **When it fires** | During generation, token by token | After full response generated |
| **Model weights** | Completely frozen | Updated via QLoRA |
| **What it reads** | Hidden state vectors (3072-dim) | The response text itself |
| **Output** | Hallucination probability score | Structured JSON verdict + correction |
| **Latency** | Very low | One additional inference call |

### Why two layers and not one?

Layer 1 gives an early warning signal but no explanation. Layer 2 gives a full verdict with correction but fires after generation. Together they cover each other's blind spots — Layer 1 catches uncertainty during generation, Layer 2 confirms and explains after. Two independent signals means much higher confidence than either alone.

<br/>

---

## Layer 2 Output Example

```json
Input:
  Question: "Who invented the telephone?"
  Answer:   "The telephone was invented by Nikola Tesla in 1876."

Output:
{
  "hallucinated": true,
  "confidence":   0.92,
  "reason":       "Telephone was invented by Alexander Graham Bell, not Tesla",
  "severity":     "high",
  "corrected":    "The telephone was invented by Alexander Graham Bell in 1876."
}
```


<br/>

---

## QLoRA Configuration

```python
# 4-bit quantisation
BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",       # NormalFloat4
    bnb_4bit_compute_dtype    = torch.bfloat16,
    bnb_4bit_use_double_quant = True
)

# LoRA adapters — only 0.17% of parameters trained
LoraConfig(
    r              = 16,
    lora_alpha     = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout   = 0.05,
    task_type      = "CAUSAL_LM"
)
```

<br/>

---

## Project Structure

```
lucid/
│
├── layer1_probing/
│   ├── extract_activations.py    # PyTorch hook setup, hidden state extraction
│   ├── train_probe.py            # Logistic regression + MLP probe training
│   └── evaluate.py               # AUROC, F1, ROC curve, PCA visualization
│
├── layer2_qlora/
│   ├── prepare_dataset.py        # HaluEval loading and formatting
│   └── train.py                  # QLoRA fine-tuning script
│
├── data/
│   ├── raw/                      # Original downloaded datasets
│   └── processed/                # train.jsonl, val.jsonl, test.jsonl
│
├── .gitignore
└── README.md
```

<br/>

---

## Setup & Usage

### Prerequisites

- Python 3.11+
- NVIDIA GPU with 6GB+ VRAM for training (or use Kaggle free tier)
- Hugging Face account with Llama 3.2 3B access approved

### Installation

```bash
git clone https://github.com/Pralishatripathy000/lucid.git
cd lucid
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
pip install transformers peft trl bitsandbytes accelerate datasets huggingface_hub wandb torch scikit-learn matplotlib
```

### Environment variables

```bash
export HF_TOKEN="your_huggingface_token"
export WANDB_API_KEY="your_wandb_key"
```

### Prepare dataset

```bash
cd layer2_qlora
python prepare_dataset.py
```

### Train Layer 2 (run on Kaggle/Colab)

```bash
python train.py
```

### Extract activations for Layer 1

```bash
cd layer1_probing
python extract_activations.py
python train_probe.py
python evaluate.py
```

<br/>

---

## Training Infrastructure

| Stage | Platform | Duration |
|---|---|---|
| Dataset preparation | Local / Kaggle | ~30 seconds |
| Layer 2 QLoRA training | Kaggle T4 x2 | ~2.5 hours (2k samples) |
| Layer 1 activation extraction | Kaggle T4 x2 | ~30 minutes |
| Layer 1 probe training | Local CPU | ~5 minutes |

Training tracked at [wandb.ai/pralishatripathy000/lucid-hallucination-detector](https://wandb.ai/pralishatripathy000-vit-bhopal/lucid-hallucination-detector)

<br/>

---

## Results

### Layer 2 — QLoRA Fine-tuned Judge (Llama 3.2 3B)

| Metric | Value |
|---|---|
| F1 Score | 0.9744 |
| AUROC | 0.9800 |
| Accuracy | 97% |
| Precision (Hallucinated) | 1.00 |
| Recall (Hallucinated) | 0.95 |
| Training samples | 2,000 |
| Epochs | 3 |
| Best eval loss | 0.629 |

### Layer 1 — Activation Probe (Frozen Llama 3.2 3B)

| Metric | Value |
|---|---|
| AUROC | 0.7785 |
| F1 Score | 0.71 |
| Accuracy | 71% |
| Samples used | 1,000 (500 hallucinated + 500 clean) |
| Layers probed | 8, 16, 24 |
| Probe architecture | MLP (256→128→64) |

## Trained Models

| Model | HuggingFace |
|---|---|
| Layer 2 — QLoRA Judge | [pralishaaaaaaaa/lucid-layer2-llama32](https://huggingface.co/pralishaaaaaaaa/lucid-layer2-llama32) |
| Layer 1 — Activation Probe | [pralishaaaaaaaa/lucid-layer1-probe](https://huggingface.co/pralishaaaaaaaa/lucid-layer1-probe) |

Training tracked at [Weights & Biases](https://wandb.ai/pralishatripathy000-vit-bhopal/lucid-hallucination-detector)


## Key Papers

- **"Language Models (Mostly) Know What They Know"** — Kadavath et al., Anthropic (2022)
- **"Probing the Internal Representations of LLMs for Truthfulness"** — Marks & Tegmark, MIT (2023)
- **"Representation Engineering"** — Zou et al. (2023)
- **"INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection"** (2024)
- **"QLoRA: Efficient Finetuning of Quantized LLMs"** — Dettmers et al. (2023)

<br/>


---

## Author

**Pralisha Tripathy**

[![GitHub](https://img.shields.io/badge/GitHub-Pralishatripathy000-181717?style=flat-square&logo=github)](https://github.com/Pralishatripathy000)
[![HuggingFace](https://img.shields.io/badge/🤗-pralishatripathy000-FFD21E?style=flat-square)](https://huggingface.co/pralishatripathy000)
[![WandB](https://img.shields.io/badge/W%26B-lucid--hallucination--detector-FFBE00?style=flat-square)](https://wandb.ai/pralishatripathy000-vit-bhopal/lucid-hallucination-detector)

<br/>

---

<div align="center">
<sub>Built from scratch. No tutorials followed. Every decision documented.</sub>
</div>
