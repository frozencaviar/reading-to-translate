"""Export the preprocessed side- and level-specific frames for the R models."""
from pathlib import Path

import pandas as pd

from run_mixed_effects_yiyu_analysis import load_data


if __name__ == "__main__":
    data = load_data(Path("attention-norm-yiyu-positive-dur"))
    frames = [
        levels[level].assign(_side=side, _level=level)
        for side, levels in data.items()
        for level in ("token", "segment")
    ]
    output = Path("model-input/all_model_data.csv")
    output.parent.mkdir(exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(output, index=False)
    print(output)
