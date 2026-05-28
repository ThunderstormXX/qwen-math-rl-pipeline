# Process

The pipeline is:

```text
Base model -> base eval -> SFT -> SFT eval -> RLVR -> RL eval -> compare
```

Base evaluation records the starting point for `Qwen/Qwen3.5-2B-Base`. SFT
warms up the model on formatted instruction/math examples. SFT evaluation
checks parseability and basic answer quality before RL starts. RLVR uses
GRPO-style rewards on math prompts. The final comparison report reads the three
evaluation JSONL files and writes Markdown plus JSON summaries.
