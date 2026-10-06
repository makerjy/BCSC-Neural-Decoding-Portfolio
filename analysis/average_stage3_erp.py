#!/usr/bin/env python3
"""
Average many stage-3 continuous EEG trials around stim_1 and plot an ERP.

The script:
1. Reads an EEGLAB .set/.fdt pair without MNE/pymatreader.
2. Selects events using their metadata fields.
3. Extracts equal-length epochs around stim_1.
4. Baseline-corrects every epoch.
5. Rejects extreme epochs using a peak-to-peak threshold.
6. Averages accepted epochs and plots mean ± SEM for one channel.

Example: average all 100 control single-vowel trials from S02
----------------------------------------------------------------
python average_stage3_erp.py \
  --set "./S02_3_filtered_data.set" \
  --task single_phoneme \
  --tms 0 \
  --target control \
  --category vowels \
  --event-role stim_1 \
  --channel FCz \
  --tmin -0.6 \
  --tmax 1.0 \
  --baseline-start -0.5 \
  --baseline-end -0.25 \
  --reject-ptp 300 \
  --output "./figures/S02_control_vowels_FCz_ERP.png" \
  --save-npz "./figures/S02_control_vowels_FCz_ERP.npz" \
  --show

Example: average only the /a/ trials
------------------------------------
Add:
  --phoneme1 a
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


@dataclass(frozen=True)
class Event:
    latency_1based: float
    event_type: str
    event_role: str
    task: str
    tms: str
    condition: str
    target: str
    category: str
    phoneme1: str
    phoneme2: str
    phoneme3: str
    trial_number: str


@dataclass
class Dataset:
    set_path: Path
    fdt_path: Path | None
    sfreq: float
    n_channels: int
    n_samples: int
    n_trials: int
    xmin: float
    channel_names: list[str]
    events: list[Event]
    embedded_data: np.ndarray | None

    @property
    def duration_seconds(self) -> float:
        return (self.n_samples - 1) / self.sfreq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an event-related potential from continuous EEGLAB stage-3 data."
    )
    parser.add_argument("--set", required=True, help="Path to the EEGLAB .set file.")

    # Event selection
    parser.add_argument("--task", default=None)
    parser.add_argument("--tms", default=None, help='For example "0" or "1".')
    parser.add_argument("--condition", default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument("--event-role", default="stim_1")
    parser.add_argument("--phoneme1", default=None)
    parser.add_argument("--phoneme2", default=None)
    parser.add_argument("--phoneme3", default=None)

    # Epoch and baseline
    parser.add_argument("--tmin", type=float, default=-0.6)
    parser.add_argument("--tmax", type=float, default=1.0)
    parser.add_argument("--baseline-start", type=float, default=-0.5)
    parser.add_argument("--baseline-end", type=float, default=-0.25)

    # Artifact rejection
    parser.add_argument(
        "--reject-ptp",
        type=float,
        default=300.0,
        help=(
            "Reject an epoch when any channel exceeds this peak-to-peak amplitude "
            "in µV. Use 0 to disable rejection. Default: 300."
        ),
    )

    # Plot
    parser.add_argument("--channel", default="FCz")
    parser.add_argument("--output", default="stage3_erp.png")
    parser.add_argument(
        "--save-npz",
        default=None,
        help="Optional output path for ERP values, SEM, times, and trial counts.",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--positive-up",
        action="store_true",
        help="Use the ordinary positive-up y-axis. Default ERP convention is negative-up.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def to_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()

    array = np.asarray(value)

    if array.ndim == 0:
        try:
            return to_text(array.item())
        except Exception:
            return str(value).strip()

    if array.dtype.kind in {"U", "S"}:
        return "".join(array.astype(str).ravel()).strip()

    if array.size == 1:
        return to_text(array.ravel()[0])

    return str(value).strip()


def to_float(value: Any, field_name: str) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(
            f"Expected one numeric value for {field_name}; got shape {array.shape}."
        )
    return float(array.ravel()[0])


def extract_eeg_dict(mat: dict[str, Any]) -> dict[str, Any]:
    """
    Supports:
    1. EEGLAB fields saved directly at the MAT-file top level.
    2. One top-level variable named EEG containing a MATLAB struct.
    """
    if "EEG" not in mat:
        return {
            key: value
            for key, value in mat.items()
            if not key.startswith("__")
        }

    eeg_array = np.asarray(mat["EEG"], dtype=object)
    eeg_object = eeg_array.ravel()[0]

    fields = getattr(eeg_object, "_fieldnames", None)
    if not fields:
        raise ValueError("The top-level EEG value is not a readable MATLAB struct.")

    return {field: getattr(eeg_object, field) for field in fields}


def read_channel_names(chanlocs: Any, n_channels: int) -> list[str]:
    if chanlocs is None:
        return [f"Ch{index + 1}" for index in range(n_channels)]

    names: list[str] = []

    for index, item in enumerate(np.atleast_1d(chanlocs).ravel()):
        label = to_text(getattr(item, "labels", ""))
        names.append(label if label else f"Ch{index + 1}")

    if len(names) < n_channels:
        names.extend(
            f"Ch{index + 1}" for index in range(len(names), n_channels)
        )

    return names[:n_channels]


def read_events(event_value: Any) -> list[Event]:
    if event_value is None:
        return []

    events: list[Event] = []

    for item in np.atleast_1d(event_value).ravel():
        if not hasattr(item, "latency"):
            continue

        try:
            latency = to_float(getattr(item, "latency"), "event.latency")
        except (TypeError, ValueError):
            continue

        events.append(
            Event(
                latency_1based=latency,
                event_type=to_text(getattr(item, "type", "")),
                event_role=to_text(getattr(item, "event_role", "")),
                task=to_text(getattr(item, "task", "")),
                tms=to_text(getattr(item, "tms", "")),
                condition=to_text(getattr(item, "condition", "")),
                target=to_text(getattr(item, "tms_target", "")),
                category=to_text(getattr(item, "category", "")),
                phoneme1=to_text(getattr(item, "phoneme1", "")),
                phoneme2=to_text(getattr(item, "phoneme2", "")),
                phoneme3=to_text(getattr(item, "phoneme3", "")),
                trial_number=to_text(getattr(item, "trial_number", "")),
            )
        )

    return events


def load_dataset(set_path: str | Path) -> Dataset:
    path = Path(set_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f".set file not found: {path}")

    try:
        mat = loadmat(
            path,
            squeeze_me=True,
            struct_as_record=False,
            chars_as_strings=True,
        )
    except NotImplementedError as exc:
        raise RuntimeError(
            "This .set appears to use MATLAB v7.3/HDF5. "
            "Resave it from EEGLAB using MATLAB v7 or v6.5 format."
        ) from exc

    eeg = extract_eeg_dict(mat)

    sfreq = to_float(eeg["srate"], "srate")
    n_channels = int(round(to_float(eeg["nbchan"], "nbchan")))
    n_samples = int(round(to_float(eeg["pnts"], "pnts")))
    n_trials = int(round(to_float(eeg.get("trials", 1), "trials")))
    xmin = to_float(eeg.get("xmin", 0.0), "xmin")

    if n_trials != 1:
        raise ValueError(
            f"Expected continuous stage-3 data with trials=1; found trials={n_trials}."
        )

    channel_names = read_channel_names(eeg.get("chanlocs"), n_channels)
    events = read_events(eeg.get("event"))

    data_field = eeg.get("data")
    embedded_data: np.ndarray | None = None
    fdt_path: Path | None = None

    if isinstance(data_field, np.ndarray) and np.issubdtype(data_field.dtype, np.number):
        embedded_data = np.asarray(data_field, dtype=np.float64).squeeze()

        if embedded_data.shape == (n_samples, n_channels):
            embedded_data = embedded_data.T

        if embedded_data.shape != (n_channels, n_samples):
            raise ValueError(
                f"Unexpected embedded data shape {embedded_data.shape}; "
                f"expected {(n_channels, n_samples)}."
            )
    else:
        fdt_name = to_text(data_field) or to_text(eeg.get("datfile"))
        if not fdt_name:
            fdt_name = path.with_suffix(".fdt").name

        candidate = Path(fdt_name)
        fdt_path = candidate if candidate.is_absolute() else path.parent / candidate

        if not fdt_path.exists():
            raise FileNotFoundError(
                "The .set metadata loaded, but the matching .fdt signal file "
                f"was not found:\n{fdt_path}"
            )

        expected_bytes = n_channels * n_samples * 4
        actual_bytes = fdt_path.stat().st_size

        if actual_bytes != expected_bytes:
            raise ValueError(
                "Unexpected .fdt file size.\n"
                f"Expected: {expected_bytes:,} bytes\n"
                f"Actual:   {actual_bytes:,} bytes\n"
                f"File:     {fdt_path}"
            )

    return Dataset(
        set_path=path,
        fdt_path=fdt_path,
        sfreq=sfreq,
        n_channels=n_channels,
        n_samples=n_samples,
        n_trials=n_trials,
        xmin=xmin,
        channel_names=channel_names,
        events=events,
        embedded_data=embedded_data,
    )


def normalize(value: str | None) -> str | None:
    return None if value is None else value.strip().casefold()


def event_matches(event: Event, args: argparse.Namespace) -> bool:
    checks = [
        (event.task, args.task),
        (event.tms, args.tms),
        (event.condition, args.condition),
        (event.target, args.target),
        (event.category, args.category),
        (event.event_role, args.event_role),
        (event.phoneme1, args.phoneme1),
        (event.phoneme2, args.phoneme2),
        (event.phoneme3, args.phoneme3),
    ]

    return all(
        expected is None or normalize(actual) == normalize(expected)
        for actual, expected in checks
    )


def open_signal(dataset: Dataset) -> np.ndarray:
    if dataset.embedded_data is not None:
        return dataset.embedded_data

    assert dataset.fdt_path is not None

    flat = np.memmap(
        dataset.fdt_path,
        dtype="<f4",
        mode="r",
        shape=(dataset.n_channels * dataset.n_samples,),
    )

    # EEGLAB/MATLAB stores the channels × time matrix in column-major order.
    # Channels vary fastest within every time sample.
    return flat.reshape(
        dataset.n_channels,
        dataset.n_samples,
        order="F",
    )


def find_channel_index(channel_names: list[str], requested: str) -> int:
    lookup = {name.casefold(): index for index, name in enumerate(channel_names)}
    index = lookup.get(requested.casefold())

    if index is None:
        raise ValueError(
            f"Channel {requested!r} was not found.\n"
            f"Available channels: {', '.join(channel_names)}"
        )

    return index


def make_epoch_offsets(
    sfreq: float,
    tmin: float,
    tmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not tmin < 0 < tmax:
        raise ValueError("Use an epoch satisfying tmin < 0 < tmax.")

    start_offset = int(round(tmin * sfreq))
    stop_offset = int(round(tmax * sfreq))

    offsets = np.arange(start_offset, stop_offset + 1, dtype=int)
    times = offsets.astype(np.float64) / sfreq
    return offsets, times


def baseline_mask(
    times: np.ndarray,
    baseline_start: float,
    baseline_end: float,
) -> np.ndarray:
    if baseline_start >= baseline_end:
        raise ValueError("baseline-start must be smaller than baseline-end.")

    mask = (times >= baseline_start) & (times <= baseline_end)

    if not np.any(mask):
        raise ValueError(
            "The baseline interval does not overlap the epoch. "
            f"Epoch: {times[0]:.3f} to {times[-1]:.3f} s; "
            f"baseline: {baseline_start:.3f} to {baseline_end:.3f} s."
        )

    return mask


def extract_and_average(
    dataset: Dataset,
    matched_events: list[Event],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, list[float]]:
    signal = open_signal(dataset)
    offsets, times = make_epoch_offsets(
        dataset.sfreq,
        args.tmin,
        args.tmax,
    )
    base_mask = baseline_mask(
        times,
        args.baseline_start,
        args.baseline_end,
    )

    accepted_epochs: list[np.ndarray] = []
    rejected_ptp_values: list[float] = []
    outside_recording = 0

    for event in matched_events:
        # EEGLAB event latency is one-based and may be fractional.
        anchor_sample = int(round(event.latency_1based - 1.0))
        sample_indices = anchor_sample + offsets

        if sample_indices[0] < 0 or sample_indices[-1] >= dataset.n_samples:
            outside_recording += 1
            continue

        epoch = np.asarray(
            signal[:, sample_indices],
            dtype=np.float64,
        )

        # Baseline-correct every channel independently.
        epoch -= np.mean(epoch[:, base_mask], axis=1, keepdims=True)

        ptp_value = float(np.max(np.ptp(epoch, axis=1)))

        if args.reject_ptp > 0 and ptp_value > args.reject_ptp:
            rejected_ptp_values.append(ptp_value)
            continue

        accepted_epochs.append(epoch)

    if not accepted_epochs:
        raise RuntimeError(
            "No epochs survived. Increase --reject-ptp, check event filters, "
            "or verify the .fdt reading order."
        )

    epochs = np.stack(accepted_epochs, axis=0)
    erp = np.mean(epochs, axis=0)

    if epochs.shape[0] > 1:
        sem = np.std(epochs, axis=0, ddof=1) / np.sqrt(epochs.shape[0])
    else:
        sem = np.zeros_like(erp)

    return (
        erp,
        sem,
        times,
        epochs.shape[0],
        outside_recording,
        rejected_ptp_values,
    )


def selection_description(args: argparse.Namespace) -> str:
    selected = [
        ("task", args.task),
        ("tms", args.tms),
        ("condition", args.condition),
        ("target", args.target),
        ("category", args.category),
        ("event", args.event_role),
        ("phoneme1", args.phoneme1),
        ("phoneme2", args.phoneme2),
        ("phoneme3", args.phoneme3),
    ]
    return ", ".join(
        f"{name}={value}"
        for name, value in selected
        if value is not None
    )


def plot_erp(
    dataset: Dataset,
    erp: np.ndarray,
    sem: np.ndarray,
    times: np.ndarray,
    channel_index: int,
    accepted_count: int,
    matched_count: int,
    args: argparse.Namespace,
) -> None:
    mean_wave = erp[channel_index]
    sem_wave = sem[channel_index]

    figure, axis = plt.subplots(figsize=(10.5, 5.8))

    axis.fill_between(
        times,
        mean_wave - sem_wave,
        mean_wave + sem_wave,
        alpha=0.22,
        label="±1 SEM",
    )
    axis.plot(
        times,
        mean_wave,
        linewidth=2.0,
        label=f"Mean ERP, n={accepted_count}",
    )

    axis.axvline(0.0, linestyle="--", linewidth=1.2, label="stim_1")
    axis.axhline(0.0, linewidth=0.8)

    axis.axvspan(
        args.baseline_start,
        args.baseline_end,
        alpha=0.10,
        label="Baseline",
    )

    axis.set_title(
        f"{dataset.set_path.stem}: {args.channel} ERP\n"
        f"{selection_description(args)} | accepted {accepted_count}/{matched_count}"
    )
    axis.set_xlabel("Time relative to stim_1 (s)")
    axis.set_ylabel("Amplitude (µV)")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)

    if not args.positive_up:
        axis.invert_yaxis()

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure.savefig(
        output_path,
        dpi=args.dpi,
        bbox_inches="tight",
    )
    print(f"\nSaved ERP figure: {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close(figure)


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.set)

    matched_events = [
        event
        for event in dataset.events
        if event_matches(event, args)
    ]

    print("\n=== Dataset ===")
    print(f"SET:       {dataset.set_path}")
    print(f"FDT:       {dataset.fdt_path or 'embedded in .set'}")
    print(f"Channels:  {dataset.n_channels}")
    print(f"SFreq:     {dataset.sfreq:g} Hz")
    print(f"Duration:  {dataset.duration_seconds:.3f} s")
    print(f"Events:    {len(dataset.events)}")

    print("\n=== ERP selection ===")
    print(selection_description(args))
    print(f"Matched events before rejection: {len(matched_events)}")

    if not matched_events:
        raise RuntimeError(
            "No events matched the requested metadata. "
            "Check spelling and capitalization of the labels."
        )

    channel_index = find_channel_index(
        dataset.channel_names,
        args.channel,
    )

    (
        erp,
        sem,
        times,
        accepted_count,
        outside_recording,
        rejected_ptp_values,
    ) = extract_and_average(
        dataset,
        matched_events,
        args,
    )

    rejected_count = len(rejected_ptp_values)

    print("\n=== Epoch result ===")
    print(f"Accepted:             {accepted_count}")
    print(f"Rejected by PTP:      {rejected_count}")
    print(f"Outside recording:    {outside_recording}")

    if rejected_ptp_values:
        print(
            "Rejected PTP range:  "
            f"{min(rejected_ptp_values):.1f}–"
            f"{max(rejected_ptp_values):.1f} µV"
        )

    plot_erp(
        dataset=dataset,
        erp=erp,
        sem=sem,
        times=times,
        channel_index=channel_index,
        accepted_count=accepted_count,
        matched_count=len(matched_events),
        args=args,
    )

    if args.save_npz:
        npz_path = Path(args.save_npz).expanduser().resolve()
        npz_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            npz_path,
            erp=erp.astype(np.float32),
            sem=sem.astype(np.float32),
            times=times.astype(np.float32),
            channel_names=np.asarray(dataset.channel_names),
            selected_channel=args.channel,
            sfreq=np.float32(dataset.sfreq),
            matched_count=np.int32(len(matched_events)),
            accepted_count=np.int32(accepted_count),
            rejected_ptp_count=np.int32(rejected_count),
            outside_recording_count=np.int32(outside_recording),
            selection=selection_description(args),
            baseline=np.asarray(
                [args.baseline_start, args.baseline_end],
                dtype=np.float32,
            ),
            epoch_window=np.asarray(
                [args.tmin, args.tmax],
                dtype=np.float32,
            ),
        )
        print(f"Saved ERP arrays: {npz_path}")


if __name__ == "__main__":
    main()
