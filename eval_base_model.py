import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
print(f"Loading {model_name}...")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

dataset = load_dataset("openai/gsm8k", "main", split="test")
correct = 0
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

    match = re.findall(r"#### (\-?[0-9\.\,]+)", response[-300:])
    if match:
        pred = match[-1].replace(",", "").replace("$", "")
        if pred == gt:
            correct += 1

    if (i+1) % 100 == 0:
        print(f"Progress: {i+1}/{total}, accuracy so far: {correct}/{i+1} = {correct/(i+1)*100:.1f}%")

print(f"\nFINAL: Base Instruct model accuracy: {correct}/{total} = {correct/total*100:.1f}%")
