import importlib.util
import sys
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[1] / "tools" / "fit_nnue_wdl_calibration.py"
SPEC = importlib.util.spec_from_file_location("fit_nnue_wdl_calibration", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_probabilities_are_normalized_and_symmetric():
    model = MODULE.Calibrator(1)
    x = torch.ones((3, 1), dtype=torch.float64)
    cp = torch.tensor([-300.0, 0.0, 300.0], dtype=torch.float64)
    probabilities = model.probabilities(x, cp)
    assert torch.allclose(
        probabilities.sum(dim=1), torch.ones(3, dtype=torch.float64)
    )
    assert torch.allclose(probabilities[0], probabilities[2].flip(0))
    assert torch.allclose(probabilities[1, 0], probabilities[1, 2])


def test_group_split_has_no_opening_leakage():
    samples = [
        MODULE.Sample(
            game_id=opening,
            opening_line=opening,
            cp=0.0,
            phase=0.5,
            ply=0.5,
            outcome=0,
        )
        for opening in range(30)
        for _ in range(2)
    ]
    train, validation, test = MODULE.group_split(samples, 7)
    groups = [
        {sample.opening_line for sample in split}
        for split in (train, validation, test)
    ]
    assert not (groups[0] & groups[1])
    assert not (groups[0] & groups[2])
    assert not (groups[1] & groups[2])
    assert set.union(*groups) == set(range(30))
