# Demo Experiment Notes

This demo run used local server artifacts only:

- Base model: `Qwen/Qwen3.5-2B-Base`
- SFT data: 1k streamed examples from `openbmb/UltraData-SFT-2605`, config `Math`, split `think`
- RL data: 500 examples from `BytedTsinghua-SIA/DAPO-Math-17k`
- SFT method: LoRA adapter SFT with TRL `SFTTrainer`
- Attention: `sdpa` fallback because `flash-attn` was not installed

Memory-safe demo command:

```bash
GPU_ID=2 \
ATTN_IMPLEMENTATION=sdpa \
SFT_MAX_SEQ_LEN=512 \
SFT_BATCH_SIZE=1 \
SFT_PACKING=false \
bash scripts/sft-training/demo/run.sh
```

The successful SFT run trained only LoRA parameters:

```text
trainable params: 10,911,744
all params: 1,892,736,832
trainable%: 0.5765
```

Training summary:

```text
train_runtime: 1804s
train_loss: 0.6099
mean_token_accuracy: 0.8221
epoch: 1
```

## Eval Snapshot

Base eval on DAPO-Math demo validation, 50 samples, 1024 max new tokens:

```text
pass_at_1: 0.06
parse_rate: 1.00
format_rate: 0.00
avg_response_length: 3182.02
```

SFT eval on DAPO-Math demo validation, 10 samples, 256 max new tokens:

```text
pass_at_1: 0.10
parse_rate: 1.00
format_rate: 0.60
avg_response_length: 500.8
repetition_collapse_rate: 0.00
```

These are not a strict apples-to-apples comparison because the sample count and
generation length differ. They are useful smoke evidence that SFT made responses
shorter and improved boxed-answer formatting.

## Prompt Comparison

Prompt:

```text
A number leaves remainder 1 when divided by 2, remainder 2 when divided by 3,
remainder 3 when divided by 4, and remainder 4 when divided by 5. What is the
smallest positive such number? Put the final answer in \boxed{}.
```

Correct answer: `59`.

Base response:

- Correctly reasoned that the number is `-1` modulo `2,3,4,5`
- Produced `\boxed{59}`
- Included a longer verification-heavy explanation

SFT response:

- Produced the same correct answer, `\boxed{59}`
- Was much shorter and closer to the desired final-answer style
- Preserved enough reasoning while reducing verbosity

Conclusion: in this demo, SFT primarily improved response style and format
control. Accuracy needs a larger, matched eval and later RLVR to assess
meaningfully.
