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

def format_prompt(messages):
    return "<bos>" + "".join(
        f"<|turn>{m['role']}\n{m['content']}<turn|>\n" for m in messages
    ) + "<|turn>assistant\n"


def main():
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    print("Gemma 4 chat. Type 'exit' or Ctrl-D to quit, 'reset' to clear history.\n")

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        if user_input.lower() == "reset":
            messages = messages[:1]
            print("(history cleared)\n")
            continue

        messages.append({"role": "user", "content": user_input})
        inputs = processor(text=format_prompt(messages), return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        outputs = model.generate(**inputs, max_new_tokens=1024)
        reply = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        reply = reply.split("<turn|>")[0].strip()
        print(f"gemma> {reply}\n")
        messages.append({"role": "assistant", "content": reply})



if __name__ == "__main__":
    main()
