#!/usr/bin/env python3
"""
Plot a long, event-centered EEG trace from a continuous EEGLAB stage-3 dataset.

This corrected version does NOT use mne.io.read_raw_eeglab().\nIt reads EEGLAB .fdt using MATLAB/Fortran column-major ordering.
It reads the MATLAB metadata with scipy.io.loadmat() and reads the paired
EEGLAB .fdt binary file directly. This bypasses pymatreader errors that can
occur with some EEGLAB structures.

Required files in the same folder:
    S02_3_filtered_data.set
    S02_3_filtered_data.fdt

Example
-------
python plot_stage3_continuous_eeg_fixed.py \
  --set "./S02_3_filtered_data.set" \
  --event-query stim_1 \
  --event-index 0 \
  --pre 0.8 \
  --post 2.5 \
  --event-lines \
  --output "./figures/S02_around_stim1_01.png" \
  --show
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt


EEG_TRACE_COLOR = "#17324d"

DEFAULT_CHANNELS = [
    "Fp1", "F3", "FC3", "C3", "Cz", "C4",
    "FC4", "P3", "Pz", "P4", "O1", "O2",
]


@dataclass(frozen=True)
class EventRecord:
    latency_samples_1based: float
    onset_seconds: float
    event_type: str
    event_role: str
    metadata: dict[str, Any]


@dataclass
class EEGLABContinuous:
    set_path: Path
    fdt_path: Path | None
    sfreq: float
    n_channels: int
    n_samples: int
    n_trials: int
    xmin: float
    channel_names: list[str]
    events: list[EventRecord]
    embedded_data: np.ndarray | None = None

    @property
    def duration_seconds(self) -> float:
        return (self.n_samples - 1) / self.sfreq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot stacked continuous EEG from an EEGLAB stage-3 .set/.fdt pair."
    )
    parser.add_argument("--set", required=True, help="Path to the EEGLAB .set file.")
    parser.add_argument(
        "--output",
        default="stage3_continuous_eeg.png",
        help="Output PNG path.",
    )
    parser.add_argument("--show", action="store_true", help="Open the figure window.")

    parser.add_argument(
        "--event-query",
        default=None,
        help=(
            "Case-insensitive substring matched against the full event type and "
            "event_role, for example stim_1 or phonemes/tms0/control/alveolar/stim_1."
        ),
    )
    parser.add_argument(
        "--event-index",
        type=int,
        default=0,
        help="0-based index among events matching --event-query.",
    )
    parser.add_argument("--pre", type=float, default=0.5)
    parser.add_argument("--post", type=float, default=1.0)

    parser.add_argument(
        "--start",
        type=float,
        default=None,
        help="Absolute recording start time in seconds; used without --event-query.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Duration for absolute-time mode.",
    )

    parser.add_argument(
        "--list-events",
        action="store_true",
        help="Print unique event types and counts, then exit.",
    )
    parser.add_argument(
        "--event-lines",
        action="store_true",
        help="Mark every event that falls inside the selected window.",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=DEFAULT_CHANNELS,
        help="Channels to plot.",
    )

    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Plot stored stage-3 values without an additional display filter.",
    )
    parser.add_argument("--lowcut", type=float, default=0.5)
    parser.add_argument("--highcut", type=float, default=40.0)

    parser.add_argument(
        "--spacing-factor",
        type=float,
        default=3.0,
        help="Increase when traces overlap; decrease when traces are too far apart.",
    )
    parser.add_argument("--line-width", type=float, default=0.75)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def _to_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    array = np.asarray(value)

    if array.ndim == 0:
        try:
            return _to_text(array.item())
        except Exception:
            return str(value)

    if array.dtype.kind in {"U", "S"}:
        return "".join(array.astype(str).ravel()).strip()

    if array.size == 1:
        return _to_text(array.ravel()[0])

    return str(value)


def _to_float(value: Any, field_name: str) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(
            f"Expected one numeric value for {field_name}, got shape {array.shape}."
        )
    return float(array.ravel()[0])


def _extract_top_level_eeg_dict(mat: dict[str, Any]) -> dict[str, Any]:
    """
    Support both common EEGLAB layouts:
      1) variables saved directly at top level using `save -struct EEG`
      2) one top-level EEG MATLAB struct
    """
    eeg_value = mat.get("EEG")

    if eeg_value is None:
        return {
            key: value
            for key, value in mat.items()
            if not key.startswith("__")
        }

    eeg_array = np.asarray(eeg_value, dtype=object)
    eeg_obj = eeg_array.ravel()[0] if eeg_array.size else eeg_value

    field_names = getattr(eeg_obj, "_fieldnames", None)
    if not field_names:
        raise ValueError("The EEG variable exists but is not a readable MATLAB struct.")

    return {name: getattr(eeg_obj, name) for name in field_names}


def _read_channel_names(chanlocs: Any, n_channels: int) -> list[str]:
    if chanlocs is None:
        return [f"Ch{index + 1}" for index in range(n_channels)]

    objects = np.atleast_1d(chanlocs).ravel()
    names: list[str] = []

    for index, item in enumerate(objects):
        label = _to_text(getattr(item, "labels", ""))
        names.append(label if label else f"Ch{index + 1}")

    if len(names) < n_channels:
        names.extend(
            f"Ch{index + 1}" for index in range(len(names), n_channels)
        )

    return names[:n_channels]


def _read_events(
    event_value: Any,
    sfreq: float,
    xmin: float,
) -> list[EventRecord]:
    if event_value is None:
        return []

    records: list[EventRecord] = []

    for item in np.atleast_1d(event_value).ravel():
        if not hasattr(item, "latency"):
            continue

        field_names = getattr(item, "_fieldnames", None) or []
        metadata = {
            field_name: getattr(item, field_name)
            for field_name in field_names
        }

        try:
            latency = _to_float(getattr(item, "latency"), "event.latency")
        except (TypeError, ValueError):
            continue

        event_type = _to_text(getattr(item, "type", ""))
        event_role = _to_text(getattr(item, "event_role", ""))

        # EEGLAB event latency is 1-based and can be fractional after resampling.
        onset = xmin + (latency - 1.0) / sfreq

        records.append(
            EventRecord(
                latency_samples_1based=latency,
                onset_seconds=onset,
                event_type=event_type,
                event_role=event_role,
                metadata=metadata,
            )
        )

    records.sort(key=lambda record: record.onset_seconds)
    return records


def load_eeglab_metadata(set_path: str | Path) -> EEGLABContinuous:
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
            "This .set file appears to use MATLAB v7.3/HDF5. "
            "Resave it from EEGLAB as MATLAB v7 or v6.5 format."
        ) from exc

    eeg = _extract_top_level_eeg_dict(mat)

    sfreq = _to_float(eeg["srate"], "srate")
    n_channels = int(round(_to_float(eeg["nbchan"], "nbchan")))
    n_samples = int(round(_to_float(eeg["pnts"], "pnts")))
    n_trials = int(round(_to_float(eeg.get("trials", 1), "trials")))
    xmin = _to_float(eeg.get("xmin", 0.0), "xmin")

    if n_trials != 1:
        raise ValueError(
            f"This script expects continuous stage-3 data (trials=1), "
            f"but this file has trials={n_trials}."
        )

    channel_names = _read_channel_names(eeg.get("chanlocs"), n_channels)
    events = _read_events(eeg.get("event"), sfreq, xmin)

    data_field = eeg.get("data")
    embedded_data: np.ndarray | None = None
    fdt_path: Path | None = None

    if isinstance(data_field, np.ndarray) and np.issubdtype(data_field.dtype, np.number):
        embedded_data = np.asarray(data_field)
        embedded_data = np.squeeze(embedded_data)

        if embedded_data.shape == (n_samples, n_channels):
            embedded_data = embedded_data.T

        if embedded_data.shape != (n_channels, n_samples):
            raise ValueError(
                "Embedded EEG.data has unexpected shape "
                f"{embedded_data.shape}; expected {(n_channels, n_samples)}."
            )
    else:
        fdt_name = _to_text(data_field) or _to_text(eeg.get("datfile"))
        if not fdt_name:
            fdt_name = path.with_suffix(".fdt").name

        candidate = Path(fdt_name)
        fdt_path = candidate if candidate.is_absolute() else path.parent / candidate

        if not fdt_path.exists():
            same_stem = path.with_suffix(".fdt")
            details = [
                "The .set metadata loaded correctly, but the signal file is missing.",
                f"Stored EEG.data reference: {fdt_name!r}",
                f"Expected path: {fdt_path}",
            ]
            if same_stem != fdt_path:
                details.append(f"Also checked same-stem path: {same_stem}")
            details.append(
                "Put the matching .fdt file in the same folder as the .set file."
            )
            raise FileNotFoundError("\n".join(details))

        expected_bytes = n_channels * n_samples * n_trials * 4
        actual_bytes = fdt_path.stat().st_size

        if actual_bytes != expected_bytes:
            raise ValueError(
                "Unexpected .fdt file size.\n"
                f"Expected {expected_bytes:,} bytes from "
                f"{n_channels} channels × {n_samples} samples × float32.\n"
                f"Actual   {actual_bytes:,} bytes: {fdt_path}"
            )

    return EEGLABContinuous(
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


def print_event_summary(dataset: EEGLABContinuous) -> None:
    if not dataset.events:
        print("No events were found.")
        return

    type_counts = Counter(event.event_type for event in dataset.events)
    role_counts = Counter(event.event_role for event in dataset.events)

    print("\n=== Event roles ===")
    for role, count in role_counts.most_common():
        print(f"{count:6d}  {role}")

    print("\n=== Full event types ===")
    for event_type, count in type_counts.most_common():
        print(f"{count:6d}  {event_type}")


def select_channel_indices(
    available: list[str],
    requested: list[str],
) -> tuple[list[int], list[str]]:
    lookup = {name.casefold(): index for index, name in enumerate(available)}

    indices: list[int] = []
    selected_names: list[str] = []
    missing: list[str] = []

    for name in requested:
        index = lookup.get(name.casefold())
        if index is None:
            missing.append(name)
        else:
            indices.append(index)
            selected_names.append(available[index])

    if missing:
        print("\n[WARNING] Missing requested channels:")
        print("  " + ", ".join(missing))

    if not indices:
        raise ValueError("None of the requested channels exist in the dataset.")

    return indices, selected_names


def find_matching_events(
    dataset: EEGLABContinuous,
    query: str,
) -> list[EventRecord]:
    target = query.casefold()

    return [
        event
        for event in dataset.events
        if target in event.event_type.casefold()
        or target in event.event_role.casefold()
    ]


def determine_window(
    dataset: EEGLABContinuous,
    args: argparse.Namespace,
) -> tuple[float, float, EventRecord | None, str]:
    recording_start = dataset.xmin
    recording_end = dataset.xmin + dataset.duration_seconds

    if args.event_query:
        if args.pre < 0 or args.post <= 0:
            raise ValueError("--pre must be >= 0 and --post must be > 0.")

        matches = find_matching_events(dataset, args.event_query)

        if not matches:
            raise ValueError(
                f"No event matched {args.event_query!r}. "
                "Run once with --list-events to inspect the labels."
            )

        if not 0 <= args.event_index < len(matches):
            raise IndexError(
                f"--event-index {args.event_index} is outside 0..{len(matches) - 1}."
            )

        anchor = matches[args.event_index]
        start = max(recording_start, anchor.onset_seconds - args.pre)
        stop = min(recording_end, anchor.onset_seconds + args.post)
        description = (
            f"{anchor.event_type} | match "
            f"{args.event_index + 1}/{len(matches)}"
        )
        return start, stop, anchor, description

    start = recording_start if args.start is None else args.start
    stop = start + args.duration

    if args.duration <= 0:
        raise ValueError("--duration must be positive.")
    if start < recording_start or start > recording_end:
        raise ValueError(
            f"--start {start} is outside recording range "
            f"{recording_start:.3f}–{recording_end:.3f} s."
        )

    stop = min(stop, recording_end)
    return start, stop, None, f"absolute time {start:.3f}–{stop:.3f} s"


def read_data_segment(
    dataset: EEGLABContinuous,
    channel_indices: list[int],
    start_seconds: float,
    stop_seconds: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    start_sample = int(
        max(
            0,
            np.floor((start_seconds - dataset.xmin) * dataset.sfreq),
        )
    )
    stop_sample = int(
        min(
            dataset.n_samples,
            np.ceil((stop_seconds - dataset.xmin) * dataset.sfreq) + 1,
        )
    )

    if stop_sample <= start_sample:
        raise ValueError("The selected window contains no samples.")

    data = open_data_matrix(dataset)
    segment = np.asarray(
        data[np.asarray(channel_indices), start_sample:stop_sample],
        dtype=np.float64,
    )

    absolute_times = (
        dataset.xmin
        + np.arange(start_sample, stop_sample, dtype=np.float64)
        / dataset.sfreq
    )
    return segment, absolute_times, start_sample, stop_sample


def open_data_matrix(dataset: EEGLABContinuous) -> np.ndarray:
    """
    Return a channels x samples view of the continuous EEG data.

    For paired .set/.fdt data this returns a NumPy memmap-backed array, so
    callers can slice only the samples needed for plotting or epoching.
    """
    if dataset.embedded_data is not None:
        return dataset.embedded_data

    assert dataset.fdt_path is not None

    # EEGLAB writes a channels x samples MATLAB matrix to .fdt.
    # MATLAB uses column-major storage, so channels vary fastest within each
    # time sample. NumPy must therefore reshape with order="F".
    flat = np.memmap(
        dataset.fdt_path,
        dtype="<f4",
        mode="r",
        shape=(dataset.n_channels * dataset.n_samples,),
    )
    return flat.reshape(
        dataset.n_channels,
        dataset.n_samples,
        order="F",
    )


def display_bandpass(
    data_uv: np.ndarray,
    sfreq: float,
    lowcut: float,
    highcut: float,
) -> np.ndarray:
    nyquist = sfreq / 2.0

    if not 0 < lowcut < highcut < nyquist:
        raise ValueError(
            "Display filter must satisfy "
            f"0 < lowcut < highcut < Nyquist ({nyquist:g} Hz)."
        )

    sos = butter(
        4,
        [lowcut, highcut],
        btype="bandpass",
        fs=sfreq,
        output="sos",
    )

    try:
        return sosfiltfilt(sos, data_uv, axis=1)
    except ValueError as exc:
        raise ValueError(
            "The chosen display window is too short for zero-phase filtering. "
            "Use a longer --pre/--post window or add --no-filter."
        ) from exc


def robust_spacing(data_uv: np.ndarray, factor: float) -> float:
    channel_ranges = (
        np.nanpercentile(data_uv, 95, axis=1)
        - np.nanpercentile(data_uv, 5, axis=1)
    )
    base = float(np.nanmedian(channel_ranges))

    if not np.isfinite(base) or base <= 0:
        base = float(np.nanstd(data_uv))

    if not np.isfinite(base) or base <= 0:
        base = 1.0

    return base * factor


def event_label_for_plot(event: EventRecord) -> str:
    return event.event_role or event.event_type.rsplit("/", 1)[-1]


def plot_stacked_eeg(
    dataset: EEGLABContinuous,
    data_uv: np.ndarray,
    absolute_times: np.ndarray,
    selected_names: list[str],
    window_start: float,
    window_stop: float,
    anchor: EventRecord | None,
    description: str,
    args: argparse.Namespace,
) -> None:
    if not args.no_filter:
        data_uv = display_bandpass(
            data_uv,
            dataset.sfreq,
            args.lowcut,
            args.highcut,
        )

    spacing = robust_spacing(data_uv, args.spacing_factor)
    offsets = np.arange(len(selected_names) - 1, -1, -1) * spacing

    if anchor is None:
        plot_times = absolute_times
        xlabel = "Recording time (s)"
    else:
        plot_times = absolute_times - anchor.onset_seconds
        xlabel = f"Time relative to {event_label_for_plot(anchor)} (s)"

    fig_height = max(7.0, 0.45 * len(selected_names) + 2.0)
    fig, ax = plt.subplots(figsize=(15.5, fig_height))

    for index, channel_name in enumerate(selected_names):
        ax.plot(
            plot_times,
            data_uv[index] + offsets[index],
            linewidth=args.line_width,
            color=EEG_TRACE_COLOR,
        )
        label_x = plot_times[0] - 0.012 * (plot_times[-1] - plot_times[0])
        ax.text(
            label_x,
            offsets[index],
            channel_name,
            ha="right",
            va="center",
            fontsize=9.5,
        )

    if anchor is not None:
        ax.axvline(0.0, linestyle="--", linewidth=1.25)
        ax.text(
            0.0,
            offsets[0] + 0.42 * spacing,
            event_label_for_plot(anchor),
            rotation=90,
            ha="right",
            va="bottom",
            fontsize=9,
        )

    if args.event_lines:
        for event in dataset.events:
            if not window_start <= event.onset_seconds <= window_stop:
                continue

            x_position = (
                event.onset_seconds
                if anchor is None
                else event.onset_seconds - anchor.onset_seconds
            )

            is_anchor = (
                anchor is not None
                and abs(event.onset_seconds - anchor.onset_seconds)
                < 0.25 / dataset.sfreq
                and event.event_type == anchor.event_type
            )
            if is_anchor:
                continue

            ax.axvline(
                x_position,
                linestyle=":",
                linewidth=0.8,
                alpha=0.7,
            )
            ax.text(
                x_position,
                offsets[0] + 0.42 * spacing,
                event_label_for_plot(event),
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    filter_text = (
        "no additional display filter"
        if args.no_filter
        else f"display filter {args.lowcut:g}–{args.highcut:g} Hz"
    )
    ax.set_title(
        f"{dataset.set_path.name}\n"
        f"{description} | {filter_text}"
    )
    ax.set_xlabel(xlabel)
    ax.set_yticks([])
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.margins(x=0.01)

    # 100 µV scale bar.
    x_range = plot_times[-1] - plot_times[0]
    scale_x = plot_times[-1] - 0.035 * x_range
    scale_bottom = offsets[-1] - 0.65 * spacing
    scale_uv = 100.0

    ax.plot(
        [scale_x, scale_x],
        [scale_bottom, scale_bottom + scale_uv],
        linewidth=2.0,
    )
    ax.text(
        scale_x + 0.007 * x_range,
        scale_bottom + scale_uv / 2,
        "100 µV",
        va="center",
        fontsize=8.5,
    )

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        dpi=args.dpi,
        bbox_inches="tight",
    )
    print(f"\nSaved figure: {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset = load_eeglab_metadata(args.set)

    print("\n=== Stage-3 dataset ===")
    print(f"SET:       {dataset.set_path}")
    print(f"FDT:       {dataset.fdt_path or 'embedded in .set'}")
    print(f"Channels:  {dataset.n_channels}")
    print(f"Samples:   {dataset.n_samples:,}")
    print(f"SFreq:     {dataset.sfreq:g} Hz")
    print(f"Duration:  {dataset.duration_seconds:.3f} s")
    print(f"Events:    {len(dataset.events):,}")

    if args.list_events:
        print_event_summary(dataset)
        return

    channel_indices, selected_names = select_channel_indices(
        dataset.channel_names,
        args.channels,
    )

    window_start, window_stop, anchor, description = determine_window(
        dataset,
        args,
    )

    print(f"\nWindow:    {window_start:.3f}–{window_stop:.3f} s")
    if anchor is not None:
        print(f"Anchor:    {anchor.event_type}")
        print(f"Onset:     {anchor.onset_seconds:.6f} s")
    print(f"Channels:  {', '.join(selected_names)}")

    data_uv, absolute_times, _, _ = read_data_segment(
        dataset,
        channel_indices,
        window_start,
        window_stop,
    )

    plot_stacked_eeg(
        dataset=dataset,
        data_uv=data_uv,
        absolute_times=absolute_times,
        selected_names=selected_names,
        window_start=window_start,
        window_stop=window_stop,
        anchor=anchor,
        description=description,
        args=args,
    )


if __name__ == "__main__":
    main()
