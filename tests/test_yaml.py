from pathlib import Path
import yaml


def test_yaml_exists():
    assert Path("detector_data.yaml").exists()


def test_yaml_required_keys():
    with open("detector_data.yaml") as f:
        config = yaml.safe_load(f)
    for key in ("path", "train", "val", "nc", "names"):
        assert key in config, f"Missing key: {key}"


def test_yaml_single_class_fish():
    with open("detector_data.yaml") as f:
        config = yaml.safe_load(f)
    assert config["nc"] == 1
    assert config["names"][0] == "fish"


def test_yaml_dataset_paths_exist():
    with open("detector_data.yaml") as f:
        config = yaml.safe_load(f)
    root = Path(config["path"])
    assert (root / config["train"]).exists(), f"Train path not found: {root / config['train']}"
    assert (root / config["val"]).exists(), f"Val path not found: {root / config['val']}"
