from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwen_sft_rlvr.data.formatting import PromptFormatter
from qwen_sft_rlvr.models.loader import ModelLoader
from qwen_sft_rlvr.models.tokenizer import TokenizerLoader


class SingleGenerator:
    def generate(self, args: argparse.Namespace) -> str:
        import torch

        prompt = self._prompt(args)
        print(f"[generate] Loading model from {args.model_path}", flush=True)
        model = ModelLoader().load(args.model_path, dtype=args.dtype)
        if torch.cuda.is_available():
            print("[generate] Moving model to cuda", flush=True)
            model = model.to("cuda")
        model.eval()
        tokenizer = TokenizerLoader().load(args.model_path)
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        kwargs = self._kwargs(args, inputs, tokenizer)
        print("[generate] Generating", flush=True)
        with torch.inference_mode():
            output = model.generate(**kwargs)
        response = tokenizer.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        self._save(args, prompt, response)
        return response

    def _prompt(self, args: argparse.Namespace) -> str:
        if args.math:
            return PromptFormatter().format_eval_math(args.prompt)
        return args.prompt

    def _kwargs(self, args: argparse.Namespace, inputs: dict, tokenizer) -> dict:
        do_sample = args.temperature > 0
        kwargs = {
            **inputs,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if do_sample:
            kwargs["temperature"] = args.temperature
            kwargs["top_p"] = args.top_p
        return kwargs

    def _save(self, args: argparse.Namespace, prompt: str, response: str) -> None:
        if not args.output:
            return
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        record = {"model_path": args.model_path, "prompt": prompt, "response": response}
        target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[generate] Wrote {target}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/base_models/Qwen3.5-2B-Base")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--math", action="store_true")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--output", default="outputs/debug/generation.json")
    return parser.parse_args()


def main() -> None:
    response = SingleGenerator().generate(parse_args())
    print("\n[response]\n" + response)


if __name__ == "__main__":
    main()
