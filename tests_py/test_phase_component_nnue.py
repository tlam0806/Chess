from __future__ import annotations

import copy
import json
import subprocess
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from nn.compact_board_data import (
    HEADER,
    canonical_architecture_input,
    pack_record,
    position_split_bucket,
)
from nn.quantized_nnue_architectures import (
    ACTIVATION_SCRELU_RELU16_ALL,
    QUANTIZATION_CONVENTION_SCALE_CLEAN,
    QUANTIZED_ARCHITECTURES,
    PhaseStackQuantizedNnueArchitecture,
)
from tools.train_phase_component_nnue import (
    AlignedComponentDataset,
    PlannedCheckpointStop,
    TotalLabelDataset,
    collate,
    make_total_loader,
    scheduled_lr,
    stockfish_score_only_wdl_error,
    train_epoch,
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


def test_shared_first_routes_gradient_to_shared_layer_and_selected_tail() -> None:
    model = PhaseStackQuantizedNnueArchitecture(
        QUANTIZED_ARCHITECTURES["A"],
        phase_layout="shared_first",
        hidden_clip=181,
        feature_weight_scale=181,
        linear_weight_scale=64,
        output_weight_scale=16,
        screlu_divisor=128,
        psqt_weight_scale=16,
        psqt_master_scale_to_cp=1000.0,
    )
    assert model.layers_for_phase(0)[0] is model.layers_for_phase(7)[0]
    assert model.layers_for_phase(0)[1] is not model.layers_for_phase(7)[1]

    # Nine physical pieces select phase 2. Its gradient must update the shared
    # 256->32 layer plus only phase 2's 32->32 tail and output.
    feature_indices = torch.arange(9, dtype=torch.long)
    offsets = torch.tensor([0], dtype=torch.long)
    prediction = model(
        feature_indices,
        offsets,
        hidden_scales=[8, 16],
        output_scale=128,
        activation=ACTIVATION_SCRELU_RELU16_ALL,
        quantization_convention=QUANTIZATION_CONVENTION_SCALE_CLEAN,
    )
    prediction.sum().backward()
    assert model.shared_hidden_layers[0].weight.grad is not None
    assert model.phase_hidden_layers[2][0].weight.grad is not None
    assert model.phase_outputs[2].weight.grad is not None
    assert model.phase_hidden_layers[1][0].weight.grad is None
    assert model.phase_hidden_layers[3][0].weight.grad is None
    assert model.phase_outputs[1].weight.grad is None
    assert model.phase_outputs[3].weight.grad is None


def test_shared_first_expands_exactly_to_independent_phase_stacks() -> None:
    kwargs = {
        "hidden_clip": 181,
        "feature_weight_scale": 181,
        "linear_weight_scale": 64,
        "output_weight_scale": 16,
        "screlu_divisor": 128,
        "psqt_weight_scale": 16,
        "psqt_master_scale_to_cp": 1000.0,
    }
    shared = PhaseStackQuantizedNnueArchitecture(
        QUANTIZED_ARCHITECTURES["A"], phase_layout="shared_first", **kwargs
    )
    expanded = PhaseStackQuantizedNnueArchitecture(
        QUANTIZED_ARCHITECTURES["A"], phase_layout="independent", **kwargs
    )
    shared_state = shared.state_dict()
    expanded_state = expanded.state_dict()
    with torch.no_grad():
        for key, destination in expanded_state.items():
            if key.startswith("phase_hidden_layers."):
                _prefix, phase, layer, suffix = key.split(".")
                if layer == "0":
                    source_key = f"shared_hidden_layers.0.{suffix}"
                else:
                    source_key = f"phase_hidden_layers.{phase}.0.{suffix}"
            else:
                source_key = key
            destination.copy_(shared_state[source_key])
    expanded.load_state_dict(expanded_state)

    # Cover multiple phase buckets in one batch.
    counts = (1, 9, 17, 29)
    offsets = torch.tensor(
        [0, counts[0], sum(counts[:2]), sum(counts[:3])], dtype=torch.long
    )
    feature_indices = torch.cat(
        [torch.arange(count, dtype=torch.long) for count in counts]
    )
    call = {
        "hidden_scales": [8, 16],
        "output_scale": 128,
        "activation": ACTIVATION_SCRELU_RELU16_ALL,
        "quantization_convention": QUANTIZATION_CONVENTION_SCALE_CLEAN,
    }
    torch.testing.assert_close(
        shared(feature_indices, offsets, **call),
        expanded(feature_indices, offsets, **call),
        rtol=0,
        atol=0,
    )


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


def test_phase_collate_accepts_split_loader_total_score() -> None:
    sample = {
        "features": [0, 1, 2],
        "aux": [0] * 13,
        "score": -417,
        "ply": 12,
        "result": -1,
    }
    _features, offsets, psqt, total = collate(
        [sample], QUANTIZED_ARCHITECTURES["A"].board_feature_count,
        torch.device("cpu"),
    )
    assert offsets.tolist() == [0]
    assert psqt.tolist() == [0.0]
    assert total.tolist() == pytest.approx([-41700.0 / 208.0])


def _board_for_phase(phase: int, variant: int) -> bytes:
    piece_count = max(2, phase * 4 + 1)
    pieces = [0] * 64
    # Vary both king placement and the remaining squares. A simple rotation only
    # creates 62 distinct boards, which is not enough to guarantee coverage of
    # both 1% CRC32 validation buckets for every phase.
    state = ((variant + 1) * 0x9E3779B1) & 0xFFFFFFFF
    state ^= state >> 16
    state = (state * 0x85EBCA6B) & 0xFFFFFFFF
    state ^= state >> 13
    friendly_king = state % 64
    state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
    enemy_king = state % 64
    if enemy_king == friendly_king:
        enemy_king = (enemy_king + 1) % 64
    pieces[friendly_king] = 6
    pieces[enemy_king] = 12
    squares = [
        square for square in range(64)
        if square not in (friendly_king, enemy_king)
    ]
    for index in range(len(squares) - 1, 0, -1):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        swap_index = state % (index + 1)
        squares[index], squares[swap_index] = squares[swap_index], squares[index]
    for index, square in enumerate(squares[: piece_count - 2]):
        pieces[square] = 1 if index % 2 == 0 else 7
    packed = bytearray(32)
    for square in range(0, 64, 2):
        packed[square // 2] = pieces[square] | (pieces[square + 1] << 4)
    return bytes(packed)


def _record_for_phase_split(phase: int, split: str, ordinal: int) -> bytes:
    for variant in range(200_000):
        board = _board_for_phase(phase, variant)
        # Restrict fixture aux state to castling bits. Random 13-bit patterns
        # can encode an impossible en-passant state and are correctly rejected
        # by the C++-parity FEN exporter.
        aux = (variant + ordinal * 7) % 16
        canonical_board, canonical_aux, _features, _aux = (
            canonical_architecture_input(
                board, aux, QUANTIZED_ARCHITECTURES["F2M"].transform
            )
        )
        bucket = position_split_bucket(canonical_board, canonical_aux, 100)
        actual = "val" if bucket == 98 else "test" if bucket == 99 else "train"
        if actual == split:
            return pack_record(board, aux, phase * 100 - 350, 20 + phase, 0)
    raise AssertionError(f"could not synthesize phase={phase} split={split}")


def test_until_overfit_trainer_writes_exportable_f2m_checkpoint(tmp_path: Path) -> None:
    records: list[bytes] = []
    for phase in range(8):
        records.extend(
            [
                _record_for_phase_split(phase, "train", 0),
                _record_for_phase_split(phase, "train", 1),
                _record_for_phase_split(phase, "val", 0),
                _record_for_phase_split(phase, "test", 0),
            ]
        )
    data = tmp_path / "phase_split.cbin"
    data.write_bytes(HEADER + b"".join(records))
    output = tmp_path / "model"
    command = [
        sys.executable,
        "tools/train_phase_nnue_until_overfit.py",
        "--data", str(data),
        "--output-dir", str(output),
        "--arch", "F2M",
        "--phase-layout", "independent",
        "--epochs", "1",
        "--patience", "1",
        "--batch-size", "8",
        "--workers", "0",
        "--eval-workers", "0",
        "--torch-threads", "1",
        "--device", "cpu",
        "--train-max-samples", "16",
        "--val-max-samples", "8",
        "--test-max-samples", "8",
        "--train-probe-max-samples", "8",
        "--shuffle-block-size", "1",
        "--lr-steps-per-epoch", "2",
        "--lr-warmup-steps", "0",
        "--epoch-peak-lrs", "0.0005",
        "--epoch-min-lrs", "0.00005",
        "--checkpoint-samples", "8",
        "--progress-batches", "0",
        "--saturation-batches", "1",
    ]
    process = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["selected_epoch"] == 1
    checkpoint = torch.load(
        output / "phase_component_best.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["architecture"] == "F2M"
    assert checkpoint["phase_stacks"] == 8
    assert checkpoint["phase_layout"] == "independent"
    assert checkpoint["metrics"]["train"]["samples"] == 16
    assert all(
        phase["samples"] > 0
        for phase in checkpoint["metrics"]["selection"]["phase"]
    )
    assert all(
        phase["samples"] > 0
        for phase in checkpoint["metrics"]["ranking"]["phase"]
    )
    verify = subprocess.run(
        [
            sys.executable,
            "tools/verify_phase_training_checkpoint.py",
            "--checkpoint", str(output / "phase_component_best.pt"),
            "--arch", "F2M",
            "--phase-layout", "independent",
            "--expected-train-samples", "16",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert verify.returncode == 0, verify.stdout + verify.stderr
    export = subprocess.run(
        [
            sys.executable,
            "tools/export_phase_quantized_nnue.py",
            "--checkpoint", str(output / "phase_component_best.pt"),
            "--output", str(output / "phase_quantized_nnue.bin"),
            "--parity-data", str(data),
            "--parity-output", str(output / "phase_quantized_nnue_parity.tsv"),
            "--parity-samples", "32",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert export.returncode == 0, export.stdout + export.stderr
    assert (output / "phase_quantized_nnue.bin").stat().st_size > 0


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


def test_in_epoch_resume_is_bit_exact() -> None:
    def make_model() -> PhaseStackQuantizedNnueArchitecture:
        torch.manual_seed(77)
        return PhaseStackQuantizedNnueArchitecture(
            QUANTIZED_ARCHITECTURES["A"],
            phase_layout="shared_first",
            hidden_clip=181,
            feature_weight_scale=181,
            linear_weight_scale=64,
            output_weight_scale=16,
            screlu_divisor=128,
            psqt_weight_scale=16,
            psqt_master_scale_to_cp=1000.0,
        )

    samples = [
        {
            "features": [0, 1, 2],
            "aux": [0] * 13,
            "psqt_score": 0,
            "positional_score": score,
        }
        for score in (-80, -40, -10, 0, 15, 35, 70, 120)
    ]
    batches = [samples[index : index + 2] for index in range(0, 8, 2)]
    args = SimpleNamespace(
        train_samples=8,
        batch_size=2,
        lr=5e-4,
        min_lr=5e-5,
        lr_warmup_steps=2,
        warmup_start_lr=0.0,
        activation=ACTIVATION_SCRELU_RELU16_ALL,
        label_mode="total",
        loss_type="huber",
        objective="total",
        target_scale=1000.0,
        huber_delta=200.0,
        wdl_exponent=2.5,
        wdl_input_offset=270.0,
        wdl_output_offset=270.0,
        wdl_input_scaling=340.0,
        wdl_output_scaling=380.0,
        progress_batches=0,
        checkpoint_samples=4,
    )

    full_model = make_model()
    full_optimizer = torch.optim.AdamW(full_model.parameters(), lr=args.lr)
    full_metrics = train_epoch(
        full_model,
        batches,  # type: ignore[arg-type]
        full_optimizer,
        torch.device("cpu"),
        [8, 16],
        128,
        args,
    )

    interrupted_model = make_model()
    interrupted_optimizer = torch.optim.AdamW(
        interrupted_model.parameters(), lr=args.lr
    )
    saved: dict[str, object] = {}

    def stop_at_checkpoint(progress: dict[str, object]) -> None:
        saved["model"] = copy.deepcopy(interrupted_model.state_dict())
        saved["optimizer"] = copy.deepcopy(interrupted_optimizer.state_dict())
        saved["progress"] = copy.deepcopy(progress)
        raise PlannedCheckpointStop

    with pytest.raises(PlannedCheckpointStop):
        train_epoch(
            interrupted_model,
            batches,  # type: ignore[arg-type]
            interrupted_optimizer,
            torch.device("cpu"),
            [8, 16],
            128,
            args,
            checkpoint_callback=stop_at_checkpoint,
        )

    resumed_model = make_model()
    resumed_model.load_state_dict(saved["model"])  # type: ignore[arg-type]
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=args.lr)
    resumed_optimizer.load_state_dict(saved["optimizer"])  # type: ignore[arg-type]
    resumed_metrics = train_epoch(
        resumed_model,
        batches[2:],  # type: ignore[arg-type]
        resumed_optimizer,
        torch.device("cpu"),
        [8, 16],
        128,
        args,
        resume_progress=saved["progress"],  # type: ignore[arg-type]
    )

    assert full_metrics["loss"] == resumed_metrics["loss"]
    for key, tensor in full_model.state_dict().items():
        torch.testing.assert_close(
            tensor,
            resumed_model.state_dict()[key],
            rtol=0,
            atol=0,
        )


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
