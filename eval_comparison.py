"""
Compare VERL-style (zero-shot + #### instruction) vs Qwen-style (4-shot) evaluation
on Qwen2.5-0.5B-Instruct for GSM8K to explain the ~30% vs 49.6% discrepancy.
"""
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


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


# 4 few-shot examples from GSM8K train split (indices 0-3)
FEW_SHOT_EXAMPLES = [
    {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "answer": "Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips altogether in April and May.\n#### 72"
    },
    {
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "answer": "Weng earns 12/60 = $0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $10.\n#### 10"
    },
    {
        "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "answer": "In the beginning, Betty has only 100 / 2 = $50.\nBetty's grandparents gave her 15 * 2 = $30.\nThis means, Betty needs 100 - 50 - 30 - 15 = $5 more.\n#### 5"
    },
    {
        "question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?",
        "answer": "Maila read 12 x 2 = 24 pages today.\nSo she was able to read a total of 12 + 24 = 36 pages since yesterday.\nThere are 120 - 36 = 84 pages left to be read.\nSince she wants to read half of the remaining pages tomorrow, then she should read 84/2 = 42 pages.\n#### 42"
    },
]

INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'


def build_zero_shot_messages(question):
    """VERL-style: zero-shot with #### instruction."""
    return [{"role": "user", "content": question + " " + INSTRUCTION}]


def build_four_shot_messages(question):
    """Qwen-style: 4-shot without #### instruction."""
    messages = []
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": ex["question"]})
        messages.append({"role": "assistant", "content": ex["answer"]})
    messages.append({"role": "user", "content": question})
    return messages


def evaluate_config(model, tokenizer, dataset, config_name, build_messages_fn):
    """Run evaluation on full dataset with given prompt config."""
    correct_strict = 0
    correct_flexible = 0
    total = len(dataset)

    print(f"\n{'='*60}")
    print(f"Config: {config_name}")
    print(f"Testing on {total} questions...")
    print(f"{'='*60}")

    for i in range(total):
        question = dataset[i]["question"]
        gt = dataset[i]["answer"].split("####")[-1].strip().replace(",", "")

        messages = build_messages_fn(question)
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

        if (i + 1) % 100 == 0:
            print(f"  [{config_name}] {i+1}/{total} | strict: {correct_strict}/{i+1} ({correct_strict/(i+1)*100:.1f}%) | flexible: {correct_flexible}/{i+1} ({correct_flexible/(i+1)*100:.1f}%)")

    print(f"\n  [{config_name}] FINAL strict:   {correct_strict}/{total} = {correct_strict/total*100:.1f}%")
    print(f"  [{config_name}] FINAL flexible: {correct_flexible}/{total} = {correct_flexible/total*100:.1f}%")
    return correct_strict, correct_flexible


if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    dataset = load_dataset("openai/gsm8k", "main", split="test")

    configs = [
        ("Zero-shot + instruction (VERL-style)", build_zero_shot_messages),
        ("4-shot, no instruction (Qwen-style)", build_four_shot_messages),
    ]

    results = {}
    for config_name, build_fn in configs:
        strict, flexible = evaluate_config(model, tokenizer, dataset, config_name, build_fn)
        results[config_name] = {"strict": strict, "flexible": flexible}

    # Summary
    total = len(dataset)
    print(f"\n{'='*60}")
    print(f"SUMMARY (out of {total} questions)")
    print(f"{'='*60}")
    for config_name, counts in results.items():
        print(f"\n{config_name}:")
        print(f"  Strict (#### regex):  {counts['strict']}/{total} = {counts['strict']/total*100:.1f}%")
        print(f"  Flexible (last num):  {counts['flexible']}/{total} = {counts['flexible']/total*100:.1f}%")

    print(f"\nReference: Qwen blog reports 49.6% for this model (4-shot eval)")
    print(f"Reference: VERL baseline page cites 49.6% from Qwen blog")
    print(f"Reference: Your VERL PPO step-0 showed ~30% (zero-shot strict)")
