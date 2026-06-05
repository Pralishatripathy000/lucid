from datasets import load_dataset
import json
import os

os.makedirs('./data/processed', exist_ok=True)

print("Loading HaluEval dataset...")
halueval = load_dataset("pminervini/HaluEval", "qa")

def format_halueval(example):
    hallucinated = {
        "instruction": "You are a hallucination detection expert. Given a question and an answer, determine if the answer contains hallucinations. Return JSON with keys: hallucinated (bool), confidence (0-1), reason (string), severity (low/medium/high), corrected (string).",
        "input": f"Question: {example['question']}\nAnswer: {example['hallucinated_answer']}",
        "output": json.dumps({"hallucinated": True, "confidence": 0.92, "reason": "Factually incorrect", "severity": "high", "corrected": example['right_answer']})
    }
    correct = {
        "instruction": "You are a hallucination detection expert. Given a question and an answer, determine if the answer contains hallucinations. Return JSON with keys: hallucinated (bool), confidence (0-1), reason (string), severity (low/medium/high), corrected (string).",
        "input": f"Question: {example['question']}\nAnswer: {example['right_answer']}",
        "output": json.dumps({"hallucinated": False, "confidence": 0.95, "reason": "Factually correct", "severity": "none", "corrected": example['right_answer']})
    }
    return [hallucinated, correct]

print("Formatting...")
all_data = [s for ex in halueval['data'] for s in format_halueval(ex)]

def save_jsonl(data, path):
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

save_jsonl(all_data[:16000], './data/processed/train.jsonl')
save_jsonl(all_data[16000:18000], './data/processed/val.jsonl')
save_jsonl(all_data[18000:], './data/processed/test.jsonl')

print(f"Train: 16000, Val: 2000, Test: {len(all_data)-18000}")