import kagglehub
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

MODEL_PATH = kagglehub.model_download("google/gemma-4/transformers/gemma-4-e4b")

# Load model
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="mps"
)

def main():
    # Prompt — manually formatted for Gemma 4 (model package is missing chat template)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a short joke about saving RAM."},
    ]
    text = "<bos>" + "".join(
        f"<|turn>{m['role']}\n{m['content']}<turn|>\n" for m in messages
    ) + "<|turn>assistant\n"

    inputs = processor(text=text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    # Generate output
    outputs = model.generate(**inputs, max_new_tokens=1024)
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    print(response)



if __name__ == "__main__":
    main()
