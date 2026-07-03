from datetime import datetime
from pathlib import Path
from typing import Optional, Union


def make_run_dir(results_dir: Union[str, Path], run_name: str, add_timestamp: bool = False) -> Path:
    results_dir = Path(results_dir)
    if add_timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{run_name}_{ts}"
    run_dir = results_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir