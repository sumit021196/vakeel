import os
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# Use Hugging Face token from environment variable
login(os.environ["HUGGINGFACE_TOKEN"])

# Load model and tokenizer
model_name = "GaganBansal/LegalContractGenerator-GPT2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_contract(contract_type, party1, party2, jurisdiction, terms):
    prompt = f"""
    Draft a {contract_type} under Indian law.
    Party 1: {party1}
    Party 2: {party2}
    Jurisdiction: {jurisdiction}
    Key Terms: {terms}
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

iface = gr.Interface(
    fn=generate_contract,
    inputs=[
        gr.Dropdown(
            label="Contract Type",
            choices=["Rent Agreement", "NDA", "Service Agreement", "Employment Contract", "Partnership Agreement"],
            value="Service Agreement"
        ),
        gr.Textbox(label="Party 1 Details (Name, Address, Role)", lines=2, placeholder="ABC Pvt Ltd, Mumbai, Employer"),
        gr.Textbox(label="Party 2 Details (Name, Address, Role)", lines=2, placeholder="John Doe, Delhi, Consultant"),
        gr.Textbox(label="Jurisdiction", placeholder="e.g., Mumbai, Maharashtra"),
        gr.Textbox(label="Key Terms (Duration, Payment, Confidentiality, etc.)", lines=4)
    ],
    outputs=gr.Textbox(label="Generated Contract", lines=25),
    title="Indian Legal Contract Generator",
    description="Generate detailed Indian legal contracts using AI. Provide the necessary party and agreement details below.",
    theme="default"
)

iface.launch()
