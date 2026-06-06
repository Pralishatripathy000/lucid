import gradio as gr
import requests
import json

API_URL = "http://localhost:8000/analyze"

def analyze(question, answer):
    if not question or not answer:
        return "Please enter both a question and an answer."
    
    try:
        response = requests.post(API_URL, json={"question": question, "answer": answer})
        result = response.json()
        
        verdict_color = {
            "HIGH CONFIDENCE HALLUCINATION": "🔴",
            "HIGH CONFIDENCE CLEAN": "🟢",
            "UNCERTAIN — FLAG FOR REVIEW": "🟡"
        }
        
        icon = verdict_color.get(result['combined_verdict'], "⚪")
        
        output = f"""
## {icon} {result['combined_verdict']}

---

### Layer 1 — Activation Probe
- **Score:** {result['layer1_score']}
- **Signal:** {result['layer1_signal']}

### Layer 2 — Fine-tuned Judge
- **Hallucinated:** {result['layer2_hallucinated']}
- **Confidence:** {result['layer2_confidence']}
- **Severity:** {result['layer2_severity']}
- **Reason:** {result['layer2_reason']}
- **Corrected Answer:** {result['layer2_corrected']}
"""
        return output

    except Exception as e:
        return f"Error: {str(e)} — Make sure the API is running at {API_URL}"

demo = gr.Interface(
    fn=analyze,
    inputs=[
        gr.Textbox(label="Question", placeholder="Who invented the telephone?"),
        gr.Textbox(label="Answer to check", placeholder="The telephone was invented by Nikola Tesla in 1876.")
    ],
    outputs=gr.Markdown(label="Lucid Analysis"),
    title="🦙 Lucid — LLM Hallucination Detector",
    description="Two-layer hallucination detection: internal activation probing + QLoRA fine-tuned judge on Llama 3.2 3B",
    examples=[
        ["Who invented the telephone?", "The telephone was invented by Nikola Tesla in 1876."],
        ["When did World War 2 end?", "World War 2 ended in 1945 with the surrender of Germany and Japan."],
        ["What is the capital of Australia?", "The capital of Australia is Sydney."],
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch(share=True)