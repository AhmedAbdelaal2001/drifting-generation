from __future__ import annotations

from pathlib import Path
import json

def load_config(path: str | Path) -> dict:
    """
    Load a config file.
    Preferred: YAML (.yaml / .yml) if PyYAML is installed.
    Fallback: JSON.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "PyYAML is required for YAML configs. Install with `pip install pyyaml` "
                "or use a JSON config."
            ) from e
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError("Config root must be a dict.")
        return cfg

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            raise ValueError("Config root must be a dict.")
        return cfg

    raise ValueError(f"Unsupported config format: {suffix} (use .yaml/.yml or .json)")