#!/usr/bin/env python3

"""
Create ERP topographic snapshots from continuous Stage-3 EEGLAB data.

Workflow
--------
1. Load Stage-3 continuous .set/.fdt
2. Find stim_1 events for the selected task
3. Extract epochs around stim_1
4. Optionally apply 0.5-40 Hz filter
5. Average trials -> ERP
6. Compute mean voltage in four time windows
7. Plot scalp topographies

Default windows
---------------
Baseline : -0.20 to 0.00 s
Early    :  0.10 to 0.20 s
Peak     :  0.35 to 0.50 s
Late     :  0.60 to 0.90 s
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mne
import numpy as np

from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt


# ============================================================
# TIME WINDOWS
# ============================================================

WINDOWS = [
    ("Baseline", -0.20, 0.00),
    ("Early",     0.10, 0.20),
    ("Peak",      0.35, 0.50),
    ("Late",      0.60, 0.90),
]


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class EventRecord:
    latency: float
    event_type: str
    event_role: str


@dataclass
class Dataset:
    set_path: Path
    sfreq: float
    n_channels: int
    n_samples: int
    channel_names: list[str]

    # x/y coordinates for plotting
    channel_positions: np.ndarray

    # whether each channel has a valid position
    valid_position: np.ndarray

    events: list[EventRecord]

    # channels x samples
    data: np.ndarray


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--set",
        required=True,
        help="Path to Stage-3 .set file",
    )

    parser.add_argument(
        "--task",
        default="phonemes",
        help="Task name. Default: phonemes",
    )

    parser.add_argument(
        "--contains",
        nargs="*",
        default=[],
        help=(
            "Additional strings that must appear in event type. "
            "Example: --contains tms0 control"
        ),
    )

    parser.add_argument(
        "--pre",
        type=float,
        default=0.5,
        help="Seconds before stim_1",
    )

    parser.add_argument(
        "--post",
        type=float,
        default=1.0,
        help="Seconds after stim_1",
    )

    parser.add_argument(
        "--lowcut",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--highcut",
        type=float,
        default=40.0,
    )

    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Do not apply an additional 0.5-40 Hz filter",
    )

    parser.add_argument(
        "--baseline-correct",
        action="store_true",
        help=(
            "Subtract -0.2 to 0 s mean from each trial/channel. "
            "Note: baseline topomap will then be approximately zero."
        ),
    )

    parser.add_argument(
        "--vlim",
        type=float,
        default=None,
        help=(
            "Fixed symmetric color limit in uV. "
            "Example: --vlim 40 gives -40 to +40 uV."
        ),
    )

    parser.add_argument(
        "--output",
        default="./figures/erp_topomaps.png",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--show",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# MATLAB HELPERS
# ============================================================

def to_text(value: Any) -> str:

    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    arr = np.asarray(value)

    if arr.size == 0:
        return ""

    if arr.ndim == 0:
        try:
            return to_text(arr.item())
        except Exception:
            return str(value)

    if arr.dtype.kind in {"U", "S"}:
        return "".join(arr.astype(str).ravel()).strip()

    if arr.size == 1:
        return to_text(arr.ravel()[0])

    return str(value)


def to_float(value: Any) -> float:

    if value is None:
        return np.nan

    arr = np.asarray(value)

    if arr.size == 0:
        return np.nan

    try:
        return float(arr.ravel()[0])
    except Exception:
        return np.nan


def get_eeg_dict(mat):

    if "EEG" not in mat:

        return {
            key: value
            for key, value in mat.items()
            if not key.startswith("__")
        }

    eeg_obj = np.asarray(mat["EEG"], dtype=object).ravel()[0]

    fields = getattr(eeg_obj, "_fieldnames", None)

    if fields is None:
        raise ValueError("Could not read EEG MATLAB structure.")

    return {
        field: getattr(eeg_obj, field)
        for field in fields
    }


# ============================================================
# CHANNEL LOCATIONS
# ============================================================

def read_channel_locations(chanlocs, n_channels):

    channel_names = []
    positions = []
    valid = []

    objects = np.atleast_1d(chanlocs).ravel()

    for ch in range(n_channels):

        if ch >= len(objects):

            channel_names.append(f"Ch{ch + 1}")
            positions.append([np.nan, np.nan])
            valid.append(False)

            continue

        item = objects[ch]

        label = to_text(
            getattr(item, "labels", "")
        )

        if not label:
            label = f"Ch{ch + 1}"

        channel_names.append(label)

        # ----------------------------------------------------
        # Preferred: EEGLAB theta / radius
        #
        # EEGLAB topographic convention:
        # theta=0 -> anterior/nose
        # ----------------------------------------------------

        theta = to_float(
            getattr(item, "theta", np.nan)
        )

        radius = to_float(
            getattr(item, "radius", np.nan)
        )

        if np.isfinite(theta) and np.isfinite(radius):

            theta_rad = np.deg2rad(theta)

            x = radius * np.sin(theta_rad)
            y = radius * np.cos(theta_rad)

            positions.append([x, y])
            valid.append(True)

            continue

        # ----------------------------------------------------
        # Fallback: Cartesian coordinates
        #
        # EEGLAB:
        # X ~ anterior
        # Y ~ left
        #
        # Convert to plotting:
        # x = left/right
        # y = posterior/anterior
        # ----------------------------------------------------

        X = to_float(
            getattr(item, "X", np.nan)
        )

        Y = to_float(
            getattr(item, "Y", np.nan)
        )

        if np.isfinite(X) and np.isfinite(Y):

            x = -Y
            y = X

            positions.append([x, y])
            valid.append(True)

        else:

            positions.append([np.nan, np.nan])
            valid.append(False)

    positions = np.asarray(
        positions,
        dtype=float,
    )

    valid = np.asarray(
        valid,
        dtype=bool,
    )

    # --------------------------------------------------------
    # Normalize coordinates to head radius ~0.48
    # --------------------------------------------------------

    if np.any(valid):

        radii = np.sqrt(
            positions[valid, 0] ** 2
            + positions[valid, 1] ** 2
        )

        max_radius = np.nanmax(radii)

        if max_radius > 0:
            positions[valid] *= (
                0.48 / max_radius
            )

    return (
        channel_names,
        positions,
        valid,
    )


# ============================================================
# EVENTS
# ============================================================

def read_events(event_value):

    events = []

    for item in np.atleast_1d(event_value).ravel():

        if not hasattr(item, "latency"):
            continue

        latency = to_float(
            getattr(item, "latency")
        )

        if not np.isfinite(latency):
            continue

        event_type = to_text(
            getattr(item, "type", "")
        )

        event_role = to_text(
            getattr(item, "event_role", "")
        )

        events.append(
            EventRecord(
                latency=latency,
                event_type=event_type,
                event_role=event_role,
            )
        )

    return events


# ============================================================
# LOAD STAGE-3
# ============================================================

def load_stage3(set_path):

    set_path = Path(
        set_path
    ).expanduser().resolve()

    if not set_path.exists():

        raise FileNotFoundError(
            f".set file not found:\n{set_path}"
        )

    print("\nLoading:")
    print(set_path)

    mat = loadmat(
        set_path,
        squeeze_me=True,
        struct_as_record=False,
        chars_as_strings=True,
    )

    eeg = get_eeg_dict(mat)

    sfreq = float(
        np.asarray(eeg["srate"]).ravel()[0]
    )

    n_channels = int(
        np.asarray(eeg["nbchan"]).ravel()[0]
    )

    n_samples = int(
        np.asarray(eeg["pnts"]).ravel()[0]
    )

    n_trials = int(
        np.asarray(
            eeg.get("trials", 1)
        ).ravel()[0]
    )

    if n_trials != 1:

        raise ValueError(
            "Stage-3 must be continuous. "
            f"trials={n_trials}"
        )

    (
        channel_names,
        channel_positions,
        valid_position,
    ) = read_channel_locations(
        eeg["chanlocs"],
        n_channels,
    )

    events = read_events(
        eeg["event"]
    )

    # ========================================================
    # READ SIGNAL
    # ========================================================

    data_field = eeg.get("data")

    if (
        isinstance(data_field, np.ndarray)
        and np.issubdtype(
            data_field.dtype,
            np.number,
        )
    ):

        data = np.asarray(
            data_field
        )

        data = np.squeeze(data)

        if data.shape == (
            n_samples,
            n_channels,
        ):
            data = data.T

    else:

        fdt_name = to_text(data_field)

        if not fdt_name:
            fdt_name = set_path.with_suffix(
                ".fdt"
            ).name

        fdt_path = set_path.parent / fdt_name

        if not fdt_path.exists():

            fdt_path = set_path.with_suffix(
                ".fdt"
            )

        if not fdt_path.exists():

            raise FileNotFoundError(
                "Matching .fdt file not found."
            )

        flat = np.memmap(
            fdt_path,
            dtype="<f4",
            mode="r",
        )

        expected = (
            n_channels
            * n_samples
        )

        if flat.size != expected:

            raise ValueError(
                f"Unexpected FDT size.\n"
                f"Expected {expected:,} float32 values\n"
                f"Found    {flat.size:,}"
            )

        data = flat.reshape(
            n_channels,
            n_samples,
            order="F",
        )

    print(
        f"Channels: {n_channels}"
    )

    print(
        f"Valid channel locations: "
        f"{np.sum(valid_position)}"
    )

    print(
        f"Sampling rate: {sfreq:g} Hz"
    )

    print(
        f"Events: {len(events)}"
    )

    return Dataset(
        set_path=set_path,
        sfreq=sfreq,
        n_channels=n_channels,
        n_samples=n_samples,
        channel_names=channel_names,
        channel_positions=channel_positions,
        valid_position=valid_position,
        events=events,
        data=data,
    )


# ============================================================
# EVENT SELECTION
# ============================================================

def select_stim1_events(
    dataset,
    task,
    contains,
):

    selected = []

    task_prefix = (
        task.casefold()
        + "/"
    )

    contains_lower = [
        item.casefold()
        for item in contains
    ]

    for event in dataset.events:

        event_type = (
            event.event_type
            .strip()
            .casefold()
        )

        event_role = (
            event.event_role
            .strip()
            .casefold()
        )

        is_task = event_type.startswith(
            task_prefix
        )

        is_stim1 = (
            event_type.endswith(
                "/stim_1"
            )
            or event_role == "stim_1"
        )

        extra_match = all(
            token in event_type
            for token in contains_lower
        )

        if (
            is_task
            and is_stim1
            and extra_match
        ):

            selected.append(event)

    return selected


# ============================================================
# FILTER
# ============================================================

def make_filter(
    sfreq,
    lowcut,
    highcut,
):

    nyquist = sfreq / 2

    if not (
        0 < lowcut
        < highcut
        < nyquist
    ):

        raise ValueError(
            "Invalid filter limits."
        )

    return butter(
        4,
        [lowcut, highcut],
        btype="bandpass",
        fs=sfreq,
        output="sos",
    )


# ============================================================
# EXTRACT EPOCHS
# ============================================================

def extract_epochs(
    dataset,
    events,
    pre,
    post,
    use_filter,
    lowcut,
    highcut,
    baseline_correct,
):

    sfreq = dataset.sfreq

    pre_n = round(
        pre * sfreq
    )

    post_n = round(
        post * sfreq
    )

    # Add padding before filtering to reduce edge effects.
    filter_pad_seconds = 1.0

    pad_n = (
        round(
            filter_pad_seconds
            * sfreq
        )
        if use_filter
        else 0
    )

    times = (
        np.arange(
            -pre_n,
            post_n + 1,
        )
        / sfreq
    )

    target_length = len(times)

    if use_filter:

        sos = make_filter(
            sfreq,
            lowcut,
            highcut,
        )

    epochs = []

    skipped = 0

    for event in events:

        # EEGLAB latency is 1-based
        center = int(
            round(
                event.latency - 1
            )
        )

        start = (
            center
            - pre_n
            - pad_n
        )

        stop = (
            center
            + post_n
            + pad_n
            + 1
        )

        if (
            start < 0
            or stop > dataset.n_samples
        ):

            skipped += 1
            continue

        segment = np.asarray(
            dataset.data[
                :,
                start:stop,
            ],
            dtype=np.float64,
        )

        if use_filter:

            segment = sosfiltfilt(
                sos,
                segment,
                axis=1,
            )

            segment = segment[
                :,
                pad_n:
                pad_n + target_length
            ]

        if baseline_correct:

            baseline_mask = (
                (times >= -0.20)
                & (times < 0)
            )

            baseline = np.mean(
                segment[
                    :,
                    baseline_mask,
                ],
                axis=1,
                keepdims=True,
            )

            segment = (
                segment - baseline
            )

        epochs.append(
            segment
        )

    if not epochs:

        raise RuntimeError(
            "No valid epochs could be extracted."
        )

    epochs = np.stack(
        epochs,
        axis=0,
    )

    print(
        f"\nValid epochs: {len(epochs)}"
    )

    if skipped:
        print(
            f"Skipped edge epochs: {skipped}"
        )

    return epochs, times


# ============================================================
# WINDOW VALUES
# ============================================================

def compute_window_maps(
    mean_erp,
    times,
):

    maps = []

    for (
        name,
        start,
        end,
    ) in WINDOWS:

        if end == 0:

            mask = (
                (times >= start)
                & (times < end)
            )

        else:

            mask = (
                (times >= start)
                & (times <= end)
            )

        if not np.any(mask):

            raise ValueError(
                f"No samples in window "
                f"{name}: {start} to {end}"
            )

        values = np.mean(
            mean_erp[
                :,
                mask,
            ],
            axis=1,
        )

        maps.append(
            values
        )

    return maps


# ============================================================
# PLOT TOPOMAPS
# ============================================================

def plot_topomaps(
    dataset,
    maps,
    n_epochs,
    task,
    contains,
    vlim,
    output,
    dpi,
    show,
    baseline_correct,
):

    valid = (
        dataset.valid_position
        & np.all(
            np.isfinite(
                dataset.channel_positions
            ),
            axis=1,
        )
    )

    if np.sum(valid) < 8:

        raise RuntimeError(
            "Too few channels have valid "
            "location information."
        )

    pos = dataset.channel_positions[
        valid
    ]

    maps_valid = [
        values[valid]
        for values in maps
    ]

    # --------------------------------------------------------
    # COMMON symmetric color scale
    # --------------------------------------------------------

    if vlim is None:

        all_values = np.concatenate(
            maps_valid
        )

        vmax = np.nanpercentile(
            np.abs(all_values),
            98,
        )

        if (
            not np.isfinite(vmax)
            or vmax == 0
        ):
            vmax = 1.0

        # round upward to nearest 5 uV
        vmax = (
            np.ceil(vmax / 5)
            * 5
        )

    else:
        vmax = abs(vlim)

    print(
        f"Topomap color range: "
        f"-{vmax:g} to +{vmax:g} µV"
    )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(15, 4.5),
    )

    last_image = None

    for ax, (
        window,
        values,
    ) in zip(
        axes,
        zip(
            WINDOWS,
            maps_valid,
        ),
    ):

        name, start, end = window

        # Newer MNE versions
        try:

            last_image, _ = (
                mne.viz.plot_topomap(
                    values,
                    pos,
                    axes=ax,
                    show=False,
                    cmap="RdBu_r",
                    vlim=(-vmax, vmax),
                    sensors=True,
                    contours=6,
                    sphere=(
                        0,
                        0,
                        0,
                        0.5,
                    ),
                    extrapolate="head",
                )
            )

        # Older MNE versions
        except TypeError:

            last_image, _ = (
                mne.viz.plot_topomap(
                    values,
                    pos,
                    axes=ax,
                    show=False,
                    cmap="RdBu_r",
                    vmin=-vmax,
                    vmax=vmax,
                    sensors=True,
                    contours=6,
                    sphere=(
                        0,
                        0,
                        0,
                        0.5,
                    ),
                    extrapolate="head",
                )
            )

        ax.set_title(
            f"{name}\n"
            f"{start:.2f} to {end:.2f} s",
            fontsize=12,
        )

    contains_text = (
        " + ".join(contains)
        if contains
        else "all conditions"
    )

    filter_note = (
        "baseline corrected"
        if baseline_correct
        else "absolute mean voltage"
    )

    fig.suptitle(
        f"ERP topographic snapshots | "
        f"task = {task} | "
        f"{contains_text} | "
        f"n = {n_epochs}\n"
        f"{filter_note}",
        fontsize=14,
    )

    cbar = fig.colorbar(
        last_image,
        ax=axes.tolist(),
        fraction=0.025,
        pad=0.04,
    )

    cbar.set_label(
        "Mean voltage (µV)",
        fontsize=11,
    )

    output = Path(
        output
    ).expanduser().resolve()

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output,
        dpi=dpi,
        bbox_inches="tight",
    )

    print(
        f"\nSaved figure:\n{output}"
    )

    # --------------------------------------------------------
    # ALSO SAVE CHANNEL VALUES
    # --------------------------------------------------------

    csv_path = output.with_suffix(
        ".csv"
    )

    header = [
        "channel"
    ] + [
        window[0]
        for window in WINDOWS
    ]

    rows = []

    valid_indices = np.where(
        valid
    )[0]

    for channel_index in valid_indices:

        row = [
            dataset.channel_names[
                channel_index
            ]
        ]

        for values in maps:

            row.append(
                values[
                    channel_index
                ]
            )

        rows.append(row)

    with open(
        csv_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            ",".join(header)
            + "\n"
        )

        for row in rows:

            f.write(
                str(row[0])
            )

            for value in row[1:]:

                f.write(
                    f",{float(value):.6f}"
                )

            f.write("\n")

    print(
        f"Saved values:\n{csv_path}"
    )

    if show:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    dataset = load_stage3(
        args.set
    )

    events = select_stim1_events(
        dataset,
        task=args.task,
        contains=args.contains,
    )

    print(
        f"\nMatched stim_1 events: "
        f"{len(events)}"
    )

    if not events:

        raise RuntimeError(
            "No matching stim_1 events found.\n"
            "Check --task and --contains."
        )

    if args.contains:

        print(
            "Required event strings: "
            + ", ".join(
                args.contains
            )
        )

    epochs, times = extract_epochs(
        dataset=dataset,
        events=events,
        pre=args.pre,
        post=args.post,
        use_filter=not args.no_filter,
        lowcut=args.lowcut,
        highcut=args.highcut,
        baseline_correct=args.baseline_correct,
    )

    # trials x channels x time
    mean_erp = np.mean(
        epochs,
        axis=0,
    )

    maps = compute_window_maps(
        mean_erp,
        times,
    )

    plot_topomaps(
        dataset=dataset,
        maps=maps,
        n_epochs=epochs.shape[0],
        task=args.task,
        contains=args.contains,
        vlim=args.vlim,
        output=args.output,
        dpi=args.dpi,
        show=args.show,
        baseline_correct=args.baseline_correct,
    )


if __name__ == "__main__":
    main()