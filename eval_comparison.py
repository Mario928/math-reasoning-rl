"""
Compare evaluation methodologies for Qwen2.5-0.5B-Instruct on GSM8K.
Config 1: Zero-shot + #### instruction (VERL-style)
Config 2: 4-shot chat template, no instruction
Config 3: 4-shot raw text, OpenCompass format ("The answer is" style)
All responses saved to JSON for inspection.
"""
import torch
import re
import json
import os
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


def extract_answer_the_answer_is(text):
    """Extract number after 'The answer is' (OpenCompass gsm8k format)."""
    matches = re.findall(r"[Tt]he answer is\s*(\-?[\$]?[0-9\.\,]+)", text)
    if matches:
        return matches[-1].replace(",", "").replace("$", "")
    return None


# 4 few-shot examples from GSM8K train split (indices 0-3) — used for Config 2
FEWSHOT_CHAT = [
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

# 4 few-shot examples from OpenCompass gsm8k_gen_1d7fe4.py — used for Config 3
# These use "The answer is X" format (not ####)
OPENCOMPASS_FEWSHOT_TEXT = """Question: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?
Let's think step by step
Answer: Angelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.
For the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.
Angelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.
However, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, so 12 total hours x 10 minutes = 120 extra minutes for breaks.
They also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.
And they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.
So Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.
They want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75
They will need to plan to study 4 days to allow for all the time they need.
The answer is 4

Question: Mark's basketball team scores 25 2 pointers, 8 3 pointers and 10 free throws.  Their opponents score double the 2 pointers but half the 3 pointers and free throws.  What's the total number of points scored by both teams added together?
Let's think step by step
Answer: Mark's team scores 25 2 pointers, meaning they scored 25*2= 50 points in 2 pointers.
His team also scores 6 3 pointers, meaning they scored 8*3= 24 points in 3 pointers
They scored 10 free throws, and free throws count as one point so they scored 10*1=10 points in free throws.
All together his team scored 50+24+10= 84 points
Mark's opponents scored double his team's number of 2 pointers, meaning they scored 50*2=100 points in 2 pointers.
His opponents scored half his team's number of 3 pointers, meaning they scored 24/2= 12 points in 3 pointers.
They also scored half Mark's team's points in free throws, meaning they scored 10/2=5 points in free throws.
All together Mark's opponents scored 100+12+5=117 points
The total score for the game is both team's scores added together, so it is 84+117=201 points
The answer is 201

Question: Bella has two times as many marbles as frisbees. She also has 20 more frisbees than deck cards. If she buys 2/5 times more of each item, what would be the total number of the items she will have if she currently has 60 marbles?
Let's think step by step
Answer: When Bella buys 2/5 times more marbles, she'll have increased the number of marbles by 2/5*60 = 24
The total number of marbles she'll have is 60+24 = 84
If Bella currently has 60 marbles, and she has two times as many marbles as frisbees, she has 60/2 = 30 frisbees.
If Bella buys 2/5 times more frisbees, she'll have 2/5*30 = 12 more frisbees.
The total number of frisbees she'll have will increase to 30+12 = 42
Bella also has 20 more frisbees than deck cards, meaning she has 30-20 = 10 deck cards
If she buys 2/5 times more deck cards, she'll have 2/5*10 = 4 more deck cards.
The total number of deck cards she'll have is 10+4 = 14
Together, Bella will have a total of 14+42+84 = 140 items
The answer is 140

Question: A group of 4 fruit baskets contains 9 apples, 15 oranges, and 14 bananas in the first three baskets and 2 less of each fruit in the fourth basket. How many fruits are there?
Let's think step by step
Answer: For the first three baskets, the number of apples and oranges in one basket is 9+15=24
In total, together with bananas, the number of fruits in one basket is 24+14=38 for the first three baskets.
Since there are three baskets each having 38 fruits, there are 3*38=114 fruits in the first three baskets.
The number of apples in the fourth basket is 9-2=7
There are also 15-2=13 oranges in the fourth basket
The combined number of oranges and apples in the fourth basket is 13+7=20
The fourth basket also contains 14-2=12 bananas.
In total, the fourth basket has 20+12=32 fruits.
The four baskets together have 32+114=146 fruits.
The answer is 146

"""

INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'


def build_zero_shot_messages(question):
    """VERL-style: zero-shot with #### instruction. Uses chat template."""
    return ("chat", [{"role": "user", "content": question + " " + INSTRUCTION}])


def build_four_shot_chat(question):
    """4-shot with chat template, no instruction."""
    messages = []
    for ex in FEWSHOT_CHAT:
        messages.append({"role": "user", "content": ex["question"]})
        messages.append({"role": "assistant", "content": ex["answer"]})
    messages.append({"role": "user", "content": question})
    return ("chat", messages)


def build_opencompass_raw(question):
    """OpenCompass format: 4-shot raw text, 'The answer is' style. No chat template."""
    prompt = OPENCOMPASS_FEWSHOT_TEXT + f"Question: {question}\nLet's think step by step\nAnswer:"
    return ("raw", prompt)


def evaluate_config(model, tokenizer, dataset, config_name, build_fn):
    """Run evaluation on full dataset with given prompt config."""
    correct_strict = 0
    correct_flexible = 0
    correct_the_answer_is = 0
    total = len(dataset)
    responses = []

    print(f"\n{'='*60}")
    print(f"Config: {config_name}")
    print(f"Testing on {total} questions...")
    print(f"{'='*60}")

    for i in range(total):
        question = dataset[i]["question"]
        gt = dataset[i]["answer"].split("####")[-1].strip().replace(",", "")

        mode, prompt_data = build_fn(question)

        if mode == "chat":
            prompt = tokenizer.apply_chat_template(prompt_data, tokenize=False, add_generation_prompt=True)
        else:
            prompt = prompt_data

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        pred_strict = extract_answer_strict(response)
        pred_flex = extract_answer_flexible(response)
        pred_tai = extract_answer_the_answer_is(response)

        is_strict = pred_strict is not None and pred_strict == gt
        is_flex = pred_flex is not None and pred_flex == gt
        is_tai = pred_tai is not None and pred_tai == gt

        if is_strict:
            correct_strict += 1
        if is_flex:
            correct_flexible += 1
        if is_tai:
            correct_the_answer_is += 1

        responses.append({
            "index": i,
            "question": question,
            "ground_truth": gt,
            "response": response,
            "pred_strict": pred_strict,
            "pred_flexible": pred_flex,
            "pred_the_answer_is": pred_tai,
            "correct_strict": is_strict,
            "correct_flexible": is_flex,
            "correct_the_answer_is": is_tai,
        })

        if (i + 1) % 100 == 0:
            print(f"  [{config_name}] {i+1}/{total} | strict: {correct_strict}/{i+1} ({correct_strict/(i+1)*100:.1f}%) | flexible: {correct_flexible}/{i+1} ({correct_flexible/(i+1)*100:.1f}%) | 'the answer is': {correct_the_answer_is}/{i+1} ({correct_the_answer_is/(i+1)*100:.1f}%)")

    print(f"\n  [{config_name}] FINAL strict:          {correct_strict}/{total} = {correct_strict/total*100:.1f}%")
    print(f"  [{config_name}] FINAL flexible:        {correct_flexible}/{total} = {correct_flexible/total*100:.1f}%")
    print(f"  [{config_name}] FINAL 'the answer is': {correct_the_answer_is}/{total} = {correct_the_answer_is/total*100:.1f}%")
    return correct_strict, correct_flexible, correct_the_answer_is, responses


if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    dataset = load_dataset("openai/gsm8k", "main", split="test")

    configs = [
        ("1_zero_shot_instruction_VERL", build_zero_shot_messages),
        ("2_four_shot_chat_template", build_four_shot_chat),
        ("3_four_shot_opencompass_raw", build_opencompass_raw),
    ]

    all_results = {}
    all_responses = {}

    for config_name, build_fn in configs:
        strict, flexible, tai, responses = evaluate_config(model, tokenizer, dataset, config_name, build_fn)
        all_results[config_name] = {"strict": strict, "flexible": flexible, "the_answer_is": tai}
        all_responses[config_name] = responses

    # Summary
    total = len(dataset)
    print(f"\n{'='*60}")
    print(f"SUMMARY (out of {total} questions)")
    print(f"{'='*60}")
    for config_name, counts in all_results.items():
        print(f"\n{config_name}:")
        print(f"  Strict (#### regex):     {counts['strict']}/{total} = {counts['strict']/total*100:.1f}%")
        print(f"  Flexible (last num):     {counts['flexible']}/{total} = {counts['flexible']/total*100:.1f}%")
        print(f"  'The answer is' regex:   {counts['the_answer_is']}/{total} = {counts['the_answer_is']/total*100:.1f}%")

    print(f"\nReference: Qwen blog reports 49.6% for this model (4-shot, in-house eval)")
    print(f"Reference: VERL baseline page cites 49.6% from Qwen blog")
    print(f"Reference: Your VERL PPO step-0 showed ~30% (zero-shot strict via vLLM)")

    # Save all responses to JSON
    output_path = "/workspace/eval_comparison_responses.json"
    with open(output_path, "w") as f:
        json.dump({"results": all_results, "responses": all_responses}, f, indent=2)
    print(f"\nAll responses saved to {output_path}")
