import torch
import json
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer

# ── Config ────────────────────────────────────────────
MODEL_NAME   = "meta-llama/Llama-3.2-3B"
OUTPUT_DIR   = "./data/activations"
NUM_SAMPLES  = 500
LAYERS       = [8, 16, 24]   # which transformer layers to probe

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load model (frozen, no training) ─────────────────
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    token=os.environ.get("HF_TOKEN")
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=os.environ.get("HF_TOKEN"))
tokenizer.pad_token = tokenizer.eos_token
print("Model loaded!")

# ── Hook setup ────────────────────────────────────────
activations = {}

def make_hook(layer_idx):
    def hook(module, input, output):
        # output is a tuple — first element is hidden state
        hidden = output[0]
        # take mean across sequence length → [batch, hidden_size]
        activations[layer_idx] = hidden.mean(dim=1).detach().cpu().float()
    return hook

# Register hooks on target layers
hooks = []
for layer_idx in LAYERS:
    hook = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
    hooks.append(hook)

# ── Load dataset ──────────────────────────────────────
print("Loading HaluEval...")
halueval = load_dataset("pminervini/HaluEval", "qa")
samples  = halueval['data'].select(range(NUM_SAMPLES))

# ── Extract activations ───────────────────────────────
print(f"Extracting activations from {NUM_SAMPLES} samples...")

all_features = []
all_labels   = []

for i, example in enumerate(samples):
    if i % 50 == 0:
        print(f"Processing {i}/{NUM_SAMPLES}...")

    for answer, label in [
        (example['hallucinated_answer'], 1),
        (example['right_answer'], 0)
    ]:
        text = f"Question: {example['question']}\nAnswer: {answer}"
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(model.device)

        with torch.no_grad():
            model(**inputs)

        # Concatenate activations from all layers
        layer_vectors = [activations[l].squeeze(0) for l in LAYERS]
        combined = torch.cat(layer_vectors, dim=0).numpy()

        all_features.append(combined.tolist())
        all_labels.append(label)

# Remove hooks
for hook in hooks:
    hook.remove()

# Save
print("Saving activations...")
data = {"features": all_features, "labels": all_labels}
with open(f"{OUTPUT_DIR}/activations.json", "w") as f:
    json.dump(data, f)

print(f"Done! Saved {len(all_labels)} samples")
print(f"Feature dimension: {len(all_features[0])}")
print(f"Hallucinated: {sum(all_labels)}, Clean: {all_labels.count(0)}")