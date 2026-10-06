#!/usr/bin/env python3
"""
Convert MATLAB-exported stage-7 files to NPZ.

This script does NOT read EEGLAB .set files. MATLAB/EEGLAB has already
exported simple arrays, so the custom EEGLAB event parsing bug is avoided.

The output NPZ contains exactly the requested keys and no extras.
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy.io import loadmat
except ImportError as exc:
    raise SystemExit(
        "Install NumPy and SciPy:\n"
        "  python -m pip install --upgrade numpy scipy"
    ) from exc

try:
    import mne
except ImportError as exc:
    raise SystemExit(
        "Install MNE for standard electrode positions in metres:\n"
        "  python -m pip install --upgrade mne"
    ) from exc


EXPECTED_KEYS = (
    "X",
    "times",
    "sfreq",
    "channel_names",
    "channel_types",
    "channel_positions_xyz",
    "channel_position_valid",
    "subject",
    "task",
    "tms_condition",
    "tms_target",
    "stimulation_site",
    "category",
    "phoneme1",
    "phoneme2",
    "phoneme3",
    "stimulus",
    "csv_row",
    "trial_n",
    "original_epoch",
    "current_epoch",
    "source_set",
    "stim1_time",
    "stim2_time",
    "stim3_time",
    "tms1_time",
    "tms2_time",
    "tms_applied",
    "stim1_in_epoch",
    "stim2_in_epoch",
    "stim3_in_epoch",
    "tms1_in_epoch",
    "tms2_in_epoch",
)

TRIAL_TEXT_FIELDS = (
    "subject",
    "task",
    "tms_condition",
    "tms_target",
    "stimulation_site",
    "category",
    "phoneme1",
    "phoneme2",
    "phoneme3",
    "stimulus",
    "source_set",
)

TRIAL_INT_FIELDS = (
    "csv_row",
    "trial_n",
    "original_epoch",
    "current_epoch",
)

TRIAL_FLOAT_FIELDS = (
    "stim1_time",
    "stim2_time",
    "stim3_time",
    "tms1_time",
    "tms2_time",
)

TRIAL_BOOL_FIELDS = (
    "tms_applied",
    "stim1_in_epoch",
    "stim2_in_epoch",
    "stim3_in_epoch",
    "tms1_in_epoch",
    "tms2_in_epoch",
)


def clean_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, bytes):
        result = value.decode("utf-8", errors="replace")
    elif isinstance(value, str):
        result = value
    elif isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        if value.dtype.kind in {"U", "S"}:
            result = "".join(value.astype(str).reshape(-1).tolist())
        else:
            result = clean_text(value.reshape(-1)[0])
    else:
        result = str(value)

    result = result.strip()

    if result.lower() in {"nan", "none", "<missing>"}:
        return ""

    return result


def string_vector(value: Any, expected_length: int) -> np.ndarray:
    array = np.asarray(value, dtype=object).reshape(-1)
    output = np.asarray(
        [clean_text(item) for item in array],
        dtype=str,
    )

    if output.size == 1 and expected_length > 1:
        output = np.full(expected_length, output[0], dtype=str)

    if output.shape != (expected_length,):
        raise ValueError(
            f"String vector has shape {output.shape}; "
            f"expected ({expected_length},)."
        )

    return output


def numeric_vector(
    value: Any,
    expected_length: int,
    dtype: np.dtype,
) -> np.ndarray:
    output = np.asarray(value).reshape(-1).astype(dtype, copy=False)

    if output.size == 1 and expected_length > 1:
        output = np.full(expected_length, output[0], dtype=dtype)

    if output.shape != (expected_length,):
        raise ValueError(
            f"Numeric vector has shape {output.shape}; "
            f"expected ({expected_length},)."
        )

    return output


def convert_raw_positions_to_metres(
    raw_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fallback only for channel names absent from the standard montage.
    """
    raw = np.asarray(raw_positions, dtype=np.float64)

    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError(
            "channel_positions_xyz_raw must have shape (channels, 3)."
        )

    valid = (
        np.all(np.isfinite(raw), axis=1)
        & (np.linalg.norm(raw, axis=1) > 0.0)
    )

    output = np.full(raw.shape, np.nan, dtype=np.float64)

    if not np.any(valid):
        return output, valid

    valid_norms = np.linalg.norm(raw[valid], axis=1)
    median_norm = float(np.median(valid_norms))

    if median_norm > 20.0:
        # Most likely millimetres.
        output[valid] = raw[valid] / 1000.0
    elif median_norm > 2.0:
        # Most likely centimetres.
        output[valid] = raw[valid] / 100.0
    elif median_norm > 0.2:
        # Unit sphere: preserve direction and use a 95-mm head radius.
        vectors = raw[valid]
        output[valid] = (
            vectors
            / np.linalg.norm(vectors, axis=1, keepdims=True)
            * 0.095
        )
    else:
        # Already metres.
        output[valid] = raw[valid]

    return output, valid


def channel_positions_in_metres(
    channel_names: np.ndarray,
    raw_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prefer MNE's standard_1005 montage, which is explicitly in metres.
    Use raw EEGLAB coordinates only as fallback for unmatched names.
    """
    montage = mne.channels.make_standard_montage("standard_1005")
    standard_positions = montage.get_positions()["ch_pos"]

    lookup = {
        name.strip().lower(): np.asarray(position, dtype=np.float64)
        for name, position in standard_positions.items()
    }

    fallback_positions, fallback_valid = (
        convert_raw_positions_to_metres(raw_positions)
    )

    positions = np.full(
        (channel_names.size, 3),
        np.nan,
        dtype=np.float64,
    )

    valid = np.zeros(channel_names.size, dtype=bool)

    for index, name in enumerate(channel_names):
        key = str(name).strip().lower()

        if key in lookup:
            positions[index] = lookup[key]
            valid[index] = True
        elif fallback_valid[index]:
            positions[index] = fallback_positions[index]
            valid[index] = True

    return positions, valid


def convert_one(
    input_path: Path,
    output_folder: Path,
    overwrite: bool,
) -> dict[str, Any]:
    output_path = output_folder / (
        input_path.name.replace("_stage7_export.mat", "")
        + "_7_ica_components_removed.npz"
    )

    if output_path.exists() and not overwrite:
        return {
            "input_mat": str(input_path),
            "output_npz": str(output_path),
            "status": "SKIPPED",
            "n_trials": "",
            "n_channels": "",
            "n_times": "",
            "message": "Output exists; use --overwrite.",
        }

    print(f"\nReading: {input_path}")

    mat = loadmat(
        input_path,
        squeeze_me=True,
        struct_as_record=False,
    )

    X = np.asarray(mat["X"], dtype=np.float32)

    if X.ndim != 3:
        raise ValueError(f"X must be 3-D, received {X.shape}.")

    n_trials, n_channels, n_times = X.shape

    if not np.all(np.isfinite(X)):
        count = int(X.size - np.count_nonzero(np.isfinite(X)))
        raise ValueError(f"X contains {count} NaN/Inf values.")

    times = np.asarray(mat["times"], dtype=np.float64).reshape(-1)
    sfreq = np.asarray(
        float(np.asarray(mat["sfreq"]).reshape(-1)[0]),
        dtype=np.float64,
    )

    if times.shape != (n_times,):
        raise ValueError(
            f"times has shape {times.shape}; expected ({n_times},)."
        )

    channel_names = string_vector(
        mat["channel_names"],
        n_channels,
    )

    channel_types = string_vector(
        mat["channel_types"],
        n_channels,
    )

    raw_positions = np.asarray(
        mat["channel_positions_xyz_raw"],
        dtype=np.float64,
    )

    if raw_positions.shape != (n_channels, 3):
        raw_positions = raw_positions.reshape(n_channels, 3)

    (
        channel_positions_xyz,
        channel_position_valid,
    ) = channel_positions_in_metres(
        channel_names,
        raw_positions,
    )

    output: dict[str, np.ndarray] = {
        "X": X,
        "times": times,
        "sfreq": sfreq,
        "channel_names": channel_names,
        "channel_types": channel_types,
        "channel_positions_xyz": channel_positions_xyz,
        "channel_position_valid": channel_position_valid,
    }

    for field_name in TRIAL_TEXT_FIELDS:
        output[field_name] = string_vector(
            mat[field_name],
            n_trials,
        )

    for field_name in TRIAL_INT_FIELDS:
        output[field_name] = numeric_vector(
            mat[field_name],
            n_trials,
            np.int32,
        )

    for field_name in TRIAL_FLOAT_FIELDS:
        output[field_name] = numeric_vector(
            mat[field_name],
            n_trials,
            np.float64,
        )

    for field_name in TRIAL_BOOL_FIELDS:
        output[field_name] = numeric_vector(
            mat[field_name],
            n_trials,
            bool,
        )

    if set(output.keys()) != set(EXPECTED_KEYS):
        missing = sorted(set(EXPECTED_KEYS) - set(output.keys()))
        extra = sorted(set(output.keys()) - set(EXPECTED_KEYS))
        raise RuntimeError(
            f"Output schema mismatch. Missing={missing}; extra={extra}"
        )

    if np.any(output["csv_row"] < 1):
        raise ValueError("At least one csv_row is missing or invalid.")

    if np.unique(output["csv_row"]).size != n_trials:
        raise ValueError("csv_row is not unique across trials.")

    output_folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **output)

    with np.load(output_path, allow_pickle=False) as saved:
        if set(saved.files) != set(EXPECTED_KEYS):
            raise RuntimeError("Saved NPZ key set changed unexpectedly.")
        if saved["X"].shape != X.shape:
            raise RuntimeError("Saved X shape changed unexpectedly.")
        if saved["X"].dtype != np.float32:
            raise RuntimeError("Saved X is not float32.")

    print(
        f"DONE: X={X.shape}, sfreq={float(sfreq):g} Hz, "
        f"positions={int(channel_position_valid.sum())}/{n_channels}"
    )
    print(f"    {output_path}")

    return {
        "input_mat": str(input_path),
        "output_npz": str(output_path),
        "status": "PROCESSED",
        "n_trials": n_trials,
        "n_channels": n_channels,
        "n_times": n_times,
        "message": "OK",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert MATLAB-exported stage-7 arrays to the exact requested "
            "NPZ schema."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Folder containing *_stage7_export.mat files.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output folder for NPZ files.",
    )

    parser.add_argument(
        "--subject",
        action="append",
        default=[],
        help="Optional filter, e.g. --subject S01 --subject P01.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing NPZ files.",
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    input_folder = args.input.expanduser().resolve()
    output_folder = args.output.expanduser().resolve()

    if not input_folder.is_dir():
        print(
            f"[ERROR] Input folder does not exist: {input_folder}",
            file=sys.stderr,
        )
        return 2

    files = sorted(input_folder.glob("*_stage7_export.mat"))

    requested = {
        value.strip().upper()
        for value in args.subject
        if value.strip()
    }

    if requested:
        files = [
            path
            for path in files
            if path.name.split("_stage7_export.mat")[0].upper()
            in requested
        ]

    if not files:
        print(
            "[ERROR] No matching *_stage7_export.mat files found.",
            file=sys.stderr,
        )
        return 2

    output_folder.mkdir(parents=True, exist_ok=True)

    rows = []
    failed = 0
    traceback_path = output_folder / "stage7_npz_traceback.txt"
    traceback_path.write_text("", encoding="utf-8")

    print(f"Found {len(files)} MATLAB export file(s).")

    for input_path in files:
        try:
            row = convert_one(
                input_path,
                output_folder,
                args.overwrite,
            )
        except Exception as exc:
            failed += 1
            message = f"{type(exc).__name__}: {exc}"
            print(f"FAILED: {message}", file=sys.stderr)

            with traceback_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    "\n"
                    + "=" * 80
                    + f"\n{input_path}\n"
                    + "=" * 80
                    + "\n"
                    + traceback.format_exc()
                    + "\n"
                )

            row = {
                "input_mat": str(input_path),
                "output_npz": "",
                "status": "FAILED",
                "n_trials": "",
                "n_channels": "",
                "n_times": "",
                "message": message,
            }

        rows.append(row)

    summary_path = output_folder / "stage7_npz_conversion_summary.csv"

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "input_mat",
                "output_npz",
                "status",
                "n_trials",
                "n_channels",
                "n_times",
                "message",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== SUMMARY ===")
    print(f"Processed: {sum(row['status'] == 'PROCESSED' for row in rows)}")
    print(f"Skipped:   {sum(row['status'] == 'SKIPPED' for row in rows)}")
    print(f"Failed:    {failed}")
    print(f"CSV:       {summary_path}")

    if failed:
        print(f"Traceback: {traceback_path}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
