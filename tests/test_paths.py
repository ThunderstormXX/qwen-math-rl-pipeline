from qwen_sft_rlvr.core.paths import RunPaths


def test_runpaths_creates_expected_directories(tmp_path):
    paths = RunPaths.from_config(
        {"paths": {"data_dir": "data", "models_dir": "models", "outputs_dir": "outputs"}},
        tmp_path,
    )
    paths.create()
    assert paths.data_dir.exists()
    assert paths.models_dir.exists()
    assert paths.outputs_dir.exists()
