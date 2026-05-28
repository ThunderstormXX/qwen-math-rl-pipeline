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


def test_deepscaler_distill_configs_load():
    teacher = ConfigLoader().load("configs/deepscaler/teacher_sft_demo.yaml")
    sft = ConfigLoader().load("scripts/deepscaler/sft-distill/demo/config.yaml")
    assert teacher.teacher.model_path.endswith("DeepScaleR-1.5B-Preview")
    assert sft.backend.name == "trl_sft"
