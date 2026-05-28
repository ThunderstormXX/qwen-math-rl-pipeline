# RL

RLVR optimizes math responses using verifiable rewards. The initial backend is
GRPO through TRL. Prompts are formatted centrally and never include ground truth.

Rewards are composed from:

- correctness against the parsed final answer
- format reward for one parseable `\boxed{...}`
- penalties for empty, long, repeated, or multi-boxed responses

DAPO-style training is represented by the planned `verl_dapo` backend. It has
the correct interface and raises `NotImplementedError` until implemented.
