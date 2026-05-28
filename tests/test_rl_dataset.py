from qwen_sft_rlvr.data.rl_dataset import RLRecordBuilder


def test_dapo_schema_builds_record():
    raw = {
        "prompt": [{"content": "Solve this.\n\nWhat is 2+2?\n\nRemember to put your answer."}],
        "reward_model": {"ground_truth": "4"},
    }
    record = RLRecordBuilder().build(raw)
    assert record is not None
    assert record["ground_truth"] == "4"
    assert "What is 2+2?" in record["problem"]
    assert "Remember to put" not in record["problem"]
