import kagglehub
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

MODEL_PATH = kagglehub.model_download("google/gemma-4/transformers/gemma-4-e4b")

# Load model
processor = AutoProcessor.from_pretrained(MODEL_PATH)
processor.chat_template = (
    "{% for m in messages %}<|turn>{{ m['role'] }}\n{{ m['content'] }}<turn|>{% endfor %}"
    "{% if add_generation_prompt %}<|turn>assistant\n{% endif %}"
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="mps"
)

def main():
    # Prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a short joke about saving RAM."},
    ]

    # Process input
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True, 
        enable_thinking=False
    )
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    # Generate output
    outputs = model.generate(**inputs, max_new_tokens=1024)
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    print(response)



if __name__ == "__main__":
    main()
