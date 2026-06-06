import os
import json
import torch
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

app = FastAPI(
    title="Lucid — LLM Hallucination Detection API",
    description="Two-layer hallucination detection: internal activation probing + QLoRA fine-tuned judge",
    version="1.0.0"
)

HF_TOKEN   = os.environ.get("HF_TOKEN")
BASE_MODEL = "meta-llama/Llama-3.2-3B"
L2_MODEL   = "./models/lucid-layer2"
L1_PROBE   = "./models/lucid-layer1/mlp_probe.pkl"
L1_SCALER  = "./models/lucid-layer1/scaler.pkl"
LAYERS     = [8, 16, 24]

# ── Request / Response schemas ────────────────────────
class AnalyzeRequest(BaseModel):
    question: str
    answer: str

class AnalyzeResponse(BaseModel):
    layer1_score:         float
    layer1_signal:        str
    layer2_hallucinated:  bool
    layer2_confidence:    float
    layer2_reason:        str
    layer2_severity:      str
    layer2_corrected:     str
    combined_verdict:     str

# ── Load models on startup ────────────────────────────
print("Loading models...")

# Base tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

# Layer 2 — fine-tuned judge
l2_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    ),
    device_map="auto",
    token=HF_TOKEN
)
l2_model = PeftModel.from_pretrained(l2_model, L2_MODEL)
l2_model.eval()

# Layer 1 — frozen base model for activation extraction
l1_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    token=HF_TOKEN
)
l1_model.eval()

# Layer 1 probe
with open(L1_PROBE, 'rb') as f:
    probe = pickle.load(f)
with open(L1_SCALER, 'rb') as f:
    scaler = pickle.load(f)

print("All models loaded!")

# ── Layer 1: activation probing ───────────────────────
def get_layer1_score(question: str, answer: str) -> float:
    activations = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            activations[layer_idx] = output[0].mean(dim=1).detach().cpu().float()
        return hook

    hooks = []
    for layer_idx in LAYERS:
        h = l1_model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    text   = f"Question: {question}\nAnswer: {answer}"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(l1_model.device)

    with torch.no_grad():
        l1_model(**inputs)

    for h in hooks:
        h.remove()

    vectors  = [activations[l].squeeze(0) for l in LAYERS]
    combined = torch.cat(vectors, dim=0).numpy().reshape(1, -1)
    scaled   = scaler.transform(combined)
    score    = probe.predict_proba(scaled)[0][1]

    return float(score)

# ── Layer 2: fine-tuned judge ─────────────────────────
def get_layer2_verdict(question: str, answer: str) -> dict:
    instruction = "You are a hallucination detection expert. Given a question and an answer, determine if the answer contains hallucinations. Return JSON with keys: hallucinated (bool), confidence (0-1), reason (string), severity (low/medium/high), corrected (string)."
    prompt = f"### Instruction:\n{instruction}\n\n### Input:\nQuestion: {question}\nAnswer: {answer}\n\n### Response:\n"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(l2_model.device)

    with torch.no_grad():
        outputs = l2_model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    try:
        result = json.loads(response.strip())
    except json.JSONDecodeError:
        result = {
            "hallucinated": True,
            "confidence": 0.5,
            "reason": "Could not parse model output",
            "severity": "unknown",
            "corrected": answer
        }

    return result

# ── Combined verdict logic ────────────────────────────
def combine_verdicts(l1_score: float, l2_result: dict) -> str:
    l2_flag = l2_result.get("hallucinated", False)
    if l1_score > 0.7 and l2_flag:
        return "HIGH CONFIDENCE HALLUCINATION"
    elif l1_score > 0.7 and not l2_flag:
        return "UNCERTAIN — FLAG FOR REVIEW"
    elif l1_score < 0.3 and l2_flag:
        return "UNCERTAIN — FLAG FOR REVIEW"
    else:
        return "HIGH CONFIDENCE CLEAN"

# ── API endpoints ─────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Lucid Hallucination Detection API", "version": "1.0.0", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    if not request.question or not request.answer:
        raise HTTPException(status_code=400, detail="Question and answer are required")

    # Layer 1
    l1_score  = get_layer1_score(request.question, request.answer)
    l1_signal = "HIGH" if l1_score > 0.7 else "LOW" if l1_score < 0.3 else "MEDIUM"

    # Layer 2
    l2_result = get_layer2_verdict(request.question, request.answer)

    # Combined
    verdict = combine_verdicts(l1_score, l2_result)

    return AnalyzeResponse(
        layer1_score        = round(l1_score, 4),
        layer1_signal       = l1_signal,
        layer2_hallucinated = l2_result.get("hallucinated", False),
        layer2_confidence   = l2_result.get("confidence", 0.0),
        layer2_reason       = l2_result.get("reason", ""),
        layer2_severity     = l2_result.get("severity", ""),
        layer2_corrected    = l2_result.get("corrected", ""),
        combined_verdict    = verdict
    )