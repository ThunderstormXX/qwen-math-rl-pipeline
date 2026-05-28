from qwen_sft_rlvr.core.config import ConfigLoader


def test_demo_sft_config_loads():
    cfg = ConfigLoader().load("scripts/sft-training/demo/config.yaml")
    assert cfg.backend.name == "trl_sft"


def test_demo_rl_config_loads():
    cfg = ConfigLoader().load("scripts/rl-training/demo/config.yaml")
    assert cfg.backend.name == "trl_grpo"


def test_required_keys_exist():
    cfg = ConfigLoader().load("scripts/sft-training/demo/config.yaml")
    ConfigLoader().require(cfg, ["run.name", "model.path", "data.train_path", "output.final_dir"])
