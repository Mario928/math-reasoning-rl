import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'


def extract_answer_strict(text):
    """Extract answer after #### (same regex as VERL's gsm8k.py)."""
    text = text[-300:] if len(text) > 300 else text
    solutions = re.findall(r"#### (\-?[0-9\.\,]+)", text)
    if solutions:
        return solutions[-1].replace(",", "").replace("$", "")
    return None


def extract_answer_flexible(text):
    """Extract last number from response (for models that skip #### format)."""
    text = text[-300:] if len(text) > 300 else text
    numbers = re.findall(r"(\-?[0-9\.\,]+)", text)
    for n in reversed(numbers):
        if n not in ["", "."]:
            return n.replace(",", "").replace("$", "")
    return None


model_name = "Qwen/Qwen2.5-0.5B-Instruct"
print(f"Loading {model_name}...")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

dataset = load_dataset("openai/gsm8k", "main", split="test")
correct_strict = 0
correct_flexible = 0
total = len(dataset)

print(f"Testing on {total} questions...")
for i in range(total):
    question = dataset[i]["question"]
    gt = dataset[i]["answer"].split("####")[-1].strip().replace(",", "")

    messages = [{"role": "user", "content": question + " " + INSTRUCTION}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    pred_strict = extract_answer_strict(response)
    if pred_strict and pred_strict == gt:
        correct_strict += 1

    pred_flex = extract_answer_flexible(response)
    if pred_flex and pred_flex == gt:
        correct_flexible += 1

    if (i+1) % 100 == 0:
        print(f"Progress: {i+1}/{total}, strict: {correct_strict}/{i+1} = {correct_strict/(i+1)*100:.1f}%, flexible: {correct_flexible}/{i+1} = {correct_flexible/(i+1)*100:.1f}%")

print(f"\nFINAL (strict, #### format, same as VERL): {correct_strict}/{total} = {correct_strict/total*100:.1f}%")
print(f"FINAL (flexible, last number): {correct_flexible}/{total} = {correct_flexible/total*100:.1f}%")
