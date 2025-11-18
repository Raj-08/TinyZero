#!/usr/bin/env python
"""
Minimal single-example Countdown evaluator.

Runs a single question from the Countdown parquet through the model
and prints:
  - the original prompt (user content),
  - the generated response,
  - the scalar reward from `countdown.compute_score`.

Usage (from TinyZero root):

  python scripts/eval_countdown_single.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --data_dir /path/to/countdown \
    --split test \
    --index 0 \
    --max_response_length 4096
"""

import argparse
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.utils.reward_score import countdown as countdown_reward


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
        "--index",
        type=int,
        default=0,
        help="Index of the example to evaluate within the split.",
    )
    parser.add_argument(
        "--max_response_length",
        type=int,
        default=4096,
        help="Max new tokens to generate for this example.",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model {args.model} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.to(device)
    model.eval()

    parquet_path = f"{args.data_dir}/{args.split}.parquet"
    print(f"Loading dataset from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    if not (0 <= args.index < len(df)):
        raise IndexError(f"Index {args.index} out of range for split of size {len(df)}")

    row = df.iloc[args.index]

    # Extract the RLHF-style prompt and convert it to a single user string,
    # mirroring RLHFDataset.__getitem__.
    chat: Any = row["prompt"]
    if isinstance(chat, np.ndarray):
        chat = chat.tolist()
    if isinstance(chat, list) and len(chat) > 0 and isinstance(chat[0], dict) and "content" in chat[0]:
        prompt_text = chat[0]["content"]
    else:
        prompt_text = str(chat)

    print("=== Prompt ===")
    print(prompt_text)
    print("==============")

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=tokenizer.model_max_length,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_response_length,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # For scoring we can feed the full sequence; the countdown scorer
    # looks for the final <answer>...</answer>. Strip extra endoftext noise.
    scored_text = full_text.replace("<|endoftext|>", "")

    print("=== Generated ===")
    print(full_text)
    print("=================")

    rm = row["reward_model"]
    ground_truth = rm["ground_truth"]
    reward = countdown_reward.compute_score(scored_text, ground_truth)

    print(f"Reward: {reward:.4f}")


if __name__ == "__main__":
    main()


