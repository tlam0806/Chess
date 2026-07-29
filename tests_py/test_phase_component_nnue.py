from __future__ import annotations

import struct
from pathlib import Path

import pytest
import torch

from nn.compact_board_data import HEADER
from nn.quantized_nnue_architectures import (
    ACTIVATION_SCRELU_RELU16_ALL,
    QUANTIZATION_CONVENTION_SCALE_CLEAN,
    QUANTIZED_ARCHITECTURES,
    PhaseStackQuantizedNnueArchitecture,
)
from tools.train_phase_component_nnue import (
    AlignedComponentDataset,
    TotalLabelDataset,
    make_total_loader,
    scheduled_lr,
    stockfish_score_only_wdl_error,
    worker_sample_limit,
)


def test_phase_bucket_routes_gradient_to_only_selected_stack() -> None:
    model = PhaseStackQuantizedNnueArchitecture(
        QUANTIZED_ARCHITECTURES["A"],
        hidden_clip=181,
        feature_weight_scale=181,
        linear_weight_scale=64,
        output_weight_scale=16,
        screlu_divisor=128,
        psqt_weight_scale=16,
        psqt_master_scale_to_cp=1000.0,
    )
    # Nine base768 features represent nine physical pieces: phase (9-1)//4=2.
    feature_indices = torch.arange(9, dtype=torch.long)
    offsets = torch.tensor([0], dtype=torch.long)
    assert model.psqt_bucket_indices(feature_indices, offsets).tolist() == [2]

    prediction = model(
        feature_indices,
        offsets,
        hidden_scales=[8, 16],
        output_scale=128,
        activation=ACTIVATION_SCRELU_RELU16_ALL,
        quantization_convention=QUANTIZATION_CONVENTION_SCALE_CLEAN,
    )
    prediction.sum().backward()
    assert model.phase_outputs[2].weight.grad is not None
    assert model.phase_outputs[1].weight.grad is None
    assert model.phase_outputs[3].weight.grad is None


def _record(score: int, first_board_byte: int = 0) -> bytes:
    board = bytearray(32)
    board[0] = 0xC6  # friendly king on a1, enemy king on b1
    board[1] = first_board_byte
    return bytes(board) + struct.pack("<HhHh", 0, score, 0, 0)


def _write(path: Path, record: bytes) -> None:
    path.write_bytes(HEADER + record)


def _write_many(path: Path, records: list[bytes]) -> None:
    path.write_bytes(HEADER + b"".join(records))


def test_aligned_component_dataset_accepts_only_score_difference(tmp_path: Path) -> None:
    psqt_dir = tmp_path / "psqt"
    positional_dir = tmp_path / "positional"
    psqt_dir.mkdir()
    positional_dir.mkdir()
    psqt = psqt_dir / "part_00000.cbin"
    positional = positional_dir / "part_00000.cbin"
    _write(psqt, _record(20))
    _write(positional, _record(-30))
    dataset = AlignedComponentDataset(psqt_dir, positional_dir, "base768", 1, 0, 1, 0)
    sample = next(iter(dataset))
    assert sample["psqt_score"] == 20
    assert sample["positional_score"] == -30


def test_aligned_component_dataset_rejects_position_mismatch(tmp_path: Path) -> None:
    psqt_dir = tmp_path / "psqt"
    positional_dir = tmp_path / "positional"
    psqt_dir.mkdir()
    positional_dir.mkdir()
    psqt = psqt_dir / "part_00000.cbin"
    positional = positional_dir / "part_00000.cbin"
    _write(psqt, _record(20, first_board_byte=0))
    _write(positional, _record(-30, first_board_byte=1))
    dataset = AlignedComponentDataset(psqt_dir, positional_dir, "base768", 1, 0, 1, 0)
    with pytest.raises(ValueError, match="component positions differ"):
        next(iter(dataset))


def test_total_label_dataset_preserves_the_single_score(tmp_path: Path) -> None:
    data_dir = tmp_path / "total"
    data_dir.mkdir()
    _write(data_dir / "part_00000.cbin", _record(-417))
    dataset = TotalLabelDataset(data_dir, "base768", 1, 0, 1, 0)
    sample = next(iter(dataset))
    assert sample["psqt_score"] == 0
    assert sample["positional_score"] == -417


@pytest.mark.parametrize(
    ("samples", "workers", "expected"),
    [
        (5_000_000, 4, [1_250_000] * 4),
        (10, 4, [3, 3, 2, 2]),
        (2, 4, [1, 1, 0, 0]),
    ],
)
def test_worker_sample_limit_is_a_global_cap(
    samples: int,
    workers: int,
    expected: list[int],
) -> None:
    limits = [worker_sample_limit(samples, worker, workers) for worker in range(workers)]
    assert limits == expected
    assert sum(limit for limit in limits if limit is not None) == samples


def test_scheduled_lr_can_warm_up_from_nonzero_resume_lr() -> None:
    values = [
        scheduled_lr(step, 20, 1e-4, 1e-5, 4, 2e-5)
        for step in range(4)
    ]
    assert values == pytest.approx([4e-5, 6e-5, 8e-5, 1e-4])


def test_total_loader_honors_global_cap_with_multiple_workers(tmp_path: Path) -> None:
    data_dir = tmp_path / "total"
    data_dir.mkdir()
    _write_many(
        data_dir / "part_00000.cbin",
        [_record(score) for score in range(20)],
    )
    loader = make_total_loader(
        data_dir,
        "base768",
        7,
        0,
        20260720,
        2,
        2,
        4,
    )
    samples = [sample for batch in loader for sample in batch]
    assert len(samples) == 7


def test_stockfish_score_only_wdl_error_matches_reference_formula() -> None:
    prediction_cp = torch.tensor([-400.0, 0.0, 750.0])
    target_cp = torch.tensor([-300.0, 100.0, 1000.0])
    actual = stockfish_score_only_wdl_error(
        prediction_cp,
        target_cp,
        exponent=2.5,
        input_offset=270.0,
        output_offset=270.0,
        input_scaling=340.0,
        output_scaling=380.0,
    )

    prediction_raw = prediction_cp * 2.08
    target_raw = target_cp * 2.08
    q = (prediction_raw - 270.0) / 340.0
    qm = (-prediction_raw - 270.0) / 340.0
    score = (target_raw - 270.0) / 380.0
    score_mirror = (-target_raw - 270.0) / 380.0
    prediction_probability = 0.5 * (
        1.0 + torch.sigmoid(q) - torch.sigmoid(qm)
    )
    target_probability = 0.5 * (
        1.0 + torch.sigmoid(score) - torch.sigmoid(score_mirror)
    )
    expected = (target_probability - prediction_probability).abs().pow(2.5)
    torch.testing.assert_close(actual, expected)
