#!/usr/bin/env python
"""
Offline evaluation script for Countdown using the same reward function
as the training pipeline, but without Ray / PPO.

Given a base model and a Countdown parquet dataset (train/test),
this script:
  - loads the tokenizer and model,
  - generates answers greedily on the val set,
  - scores them with `verl.utils.reward_score.countdown.compute_score`,
  - reports mean reward and strict accuracy.

Usage (from TinyZero root):

  python scripts/eval_countdown_model.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --data_dir /path/to/countdown \
    --split test \
    --max_response_length 4096

The parquet file is expected at {data_dir}/{split}.parquet and should
be produced by `examples/data_preprocess/countdown.py`.
"""

import argparse
from typing import List, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np

from verl.utils.reward_score import countdown as countdown_reward


def generate_responses(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    device: torch.device,
) -> List[str]:
    """Generate completions for a batch of string prompts."""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
    ).to(device)

    with torch.no_grad():
        # If model is wrapped in DataParallel, call generate on the underlying module.
        generate_target = model.module if isinstance(model, torch.nn.DataParallel) else model
        outputs = generate_target.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode full sequences (prompt + response). The countdown reward
    # function is robust to prefixes and looks for the final <answer>...</answer>.
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    return responses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Base model name or path (e.g., deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing Countdown parquet files (train.parquet / test.parquet)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Which split to evaluate: 'train' or 'test'. Default: test",
    )
    parser.add_argument(
        "--max_response_length",
        type=int,
        default=4096,
        help="Max new tokens to generate per example (should match training).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for generation.",
    )
    parser.add_argument(
        "--eval_size",
        type=int,
        default=None,
        help="If set, evaluate only the first N examples of the split.",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model {args.model} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)

    # Use all available GPUs for faster batched generation.
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel over {torch.cuda.device_count()} GPUs for inference.")
        model = torch.nn.DataParallel(model)

    model.to(device)
    model.eval()

    parquet_path = f"{args.data_dir}/{args.split}.parquet"
    print(f"Loading dataset from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    # Prompts are stored in RLHF format (list of messages with 'content').
    raw_prompts = df["prompt"].tolist()
    reward_model_data = df["reward_model"].tolist()

    # Optionally subsample the first N examples for faster eval.
    if args.eval_size is not None:
        total = min(len(raw_prompts), args.eval_size)
        raw_prompts = raw_prompts[:total]
        reward_model_data = reward_model_data[:total]
    else:
        total = len(raw_prompts)
    total_reward = 0.0
    total_correct = 0

    print(f"Evaluating {total} examples...")
    for start in range(0, total, args.batch_size):
        end = min(start + args.batch_size, total)

        # Normalize prompts to List[str] for the tokenizer.
        # We mimic RLHFDataset.__getitem__: it takes `chat = row[self.prompt_key]`
        # (which is often a numpy array of message dicts) and then uses chat[0]['content'].
        batch_raw_prompts = raw_prompts[start:end]
        batch_prompts: List[str] = []
        for p in batch_raw_prompts:
            chat = p
            # Convert numpy arrays that store the chat messages to Python lists
            if isinstance(chat, np.ndarray):
                chat = chat.tolist()
            if isinstance(chat, list) and len(chat) > 0 and isinstance(chat[0], dict) and "content" in chat[0]:
                batch_prompts.append(chat[0]["content"])
            else:
                # Fallback: string representation so tokenizer always sees a str
                batch_prompts.append(str(chat))

        batch_rm = reward_model_data[start:end]

        batch_responses = generate_responses(
            model=model,
            tokenizer=tokenizer,
            prompts=batch_prompts,
            max_new_tokens=args.max_response_length,
            device=device,
        )

        for resp, rm in zip(batch_responses, batch_rm):
            gt = rm["ground_truth"]
            # Use the same reward function as training.
            r = countdown_reward.compute_score(resp, gt)
            total_reward += float(r)
            if r >= 0.99:
                total_correct += 1

    mean_reward = total_reward / total if total > 0 else 0.0
    accuracy = total_correct / total if total > 0 else 0.0

    print(f"Mean reward: {mean_reward:.4f}")
    print(f"Accuracy (score==1.0): {accuracy:.4f}")


if __name__ == "__main__":
    main()


