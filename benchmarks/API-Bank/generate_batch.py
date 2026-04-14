"""
generate_batch.py — batched replacement for generate.py
Sends all remaining questions to vLLM in one batch instead of one-by-one.
Resumes from existing result.json (skips already-done questions).
Usage: python generate_batch.py --model_paths <path>
       WORLD_SIZE env var controls tensor_parallel_size (default 1)
"""
import os
import json
import argparse
from vllm import LLM, SamplingParams
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_paths", type=str, required=True)
    args = parser.parse_args()

    model_paths = [p.strip() for p in args.model_paths.split(",") if p.strip()]

    for model_path in model_paths:
        model_name = model_path.split("TinyZero/")[-1].replace("/", "_")
        save_path = f"PATH_TO_YOUR_SCORE_ROOT/{model_name}"
        os.makedirs(save_path, exist_ok=True)
        result_save_path = os.path.join(save_path, "result.json")

        # Load existing results (resume)
        results = json.load(open(result_save_path)) if os.path.exists(result_save_path) else {}
        print(f"Loaded {len(results)} existing results, will skip these.")

        # Collect all remaining questions
        pending_keys = []
        pending_messages = []
        pending_data = []

        for level in ["1", "2", "3"]:
            data_path = f"./level-{level}-api_processed.json"
            datas = json.load(open(data_path))
            for id, data in enumerate(datas):
                gold = f"Level{level}_{id}"
                if gold in results:
                    continue  # already done
                messages = [
                    {"role": "system", "content": data["system"]},
                    {"role": "user",   "content": data["user"]},
                ]
                pending_keys.append(gold)
                pending_messages.append(messages)
                pending_data.append(data)

        print(f"Remaining: {len(pending_messages)} questions to generate.")

        if not pending_messages:
            print("All done already.")
            continue

        # Load model once
        print("Loading model with vLLM...")
        llm = LLM(
            model=model_path,
            tensor_parallel_size=int(os.getenv("WORLD_SIZE", 1)),
            gpu_memory_utilization=0.85,   # use more GPU memory → bigger KV cache → better batching
            max_model_len=4096,
        )
        sampling_params = SamplingParams(max_tokens=4096, temperature=0.0001)

        # Single batched call — vLLM handles scheduling internally
        print(f"Running batched inference on {len(pending_messages)} questions...")
        outputs = llm.chat(pending_messages, sampling_params=sampling_params, use_tqdm=True)

        # Parse and save all results
        log = {"success": 0, "fail": 0}
        for gold, data, output in zip(pending_keys, pending_data, outputs):
            try:
                assistant_output = output.outputs[0].text.strip()
                thought = assistant_output.split("<think>")[-1].split("</think>")[0].strip()
                tool_calls_raw = assistant_output.split("<tool_call>")[-1].split("</tool_call>")[0].strip()
                all_tool_calls = []
                for tc in tool_calls_raw.strip().split("\n"):
                    if tc.strip():
                        try:
                            all_tool_calls.append(json.loads(tc))
                        except:
                            pass
                results[gold] = {
                    "data": data,
                    "raw_output": assistant_output,
                    "thought": thought,
                    "tool_calls": all_tool_calls,
                }
                log["success"] += 1
            except Exception as e:
                print(f"Error parsing {gold}: {e}")
                results[gold] = {"data": data, "raw_output": "", "thought": "", "tool_calls": []}
                log["fail"] += 1

        with open(result_save_path, "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        print(f"Done. {log}")
        print(f"Saved {len(results)} total results to {result_save_path}")
