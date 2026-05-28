# Eval

Evaluation is independent from training. It loads a checkpoint, generates
responses for eval prompts, writes per-sample generations as JSONL, and writes
aggregate metrics as JSON plus JSONL.

Metrics:

- `pass_at_1`: fraction of correct generated answers
- `parse_rate`: fraction with a parsed answer
- `format_rate`: fraction with one parseable boxed answer
- `exact_match`: normalized string exact match
- `avg_response_length`: mean generated character count
- `invalid_answer_rate`: `1 - parse_rate`
- `repetition_collapse_rate`: repeated-token pattern rate

SFT gates prevent RL from starting if parse rate is too low, invalid answers are
too high, average responses are too long, or severe repetition collapse is
detected by inspection.
