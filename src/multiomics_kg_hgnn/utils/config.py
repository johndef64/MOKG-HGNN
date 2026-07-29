"""Minimal YAML config loader (vendored so the framework has no dependency on
the MOGNN-TF benchmark package)."""
import yaml


def load_config(file_path: str) -> dict:
    """Load a YAML config file into a plain Python dict."""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)
