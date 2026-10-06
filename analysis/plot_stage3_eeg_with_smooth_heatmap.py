#!/usr/bin/env python3
"""
Plot a stimulus-centered continuous EEG overview with smoother per-channel
spectrograms styled more like the reference figure.

Layout
------
Left  : smooth spectrogram/heatmap for each selected channel (NOT averaged)
Right : stacked continuous EEG traces for the same time window

Key differences from the previous version:
- one heatmap per selected channel (no averaging across channels)
- smoother rendering via Gaussian smoothing + bilinear interpolation
- optional dashed white contour overlay per channel
- default channel set updated to Fp1, F3, C4, O1
- default frequency range capped at 40 Hz
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import spectrogram

from plot_stage3_continuous_eeg_2 import (
    EEG_TRACE_COLOR,
    EEGLABContinuous,
    EventRecord,
    _to_text,
    display_bandpass,
    load_eeglab_metadata,
    read_data_segment,
    robust_spacing,
    select_channel_indices,
)

DEFAULT_OVERVIEW_CHANNELS = [
    "Fp1", "F3", "FC3", "C3", "Cz", "C4",
    "FC4", "P3", "Pz", "P4", "O1", "O2",
]
DEFAULT_SPECTROGRAM_CHANNELS = ["Fp1", "F3", "C4", "O1"]

STIM_COLOR = "#111111"
PRE_BG = "#eef2f6"
POST_BG = "#fff1dc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot stimulus-centered continuous EEG with smooth per-channel "
            "spectrograms from an EEGLAB stage-3 .set/.fdt pair."
        )
    )

    parser.add_argument("--set", required=True, help="Path to stage-3 .set file.")
    parser.add_argument(
        "--output",
        default="eeg_with_smooth_multispec.png",
        help="Output image path (.png recommended).",
    )
    parser.add_argument("--show", action="store_true")

    # Match metadata-based anchor selection used by figure1.py.
    parser.add_argument("--task", default="single_phoneme")
    parser.add_argument("--tms", default="0")
    parser.add_argument("--target", default="control")
    parser.add_argument("--category", default="vowels")
    parser.add_argument("--anchor-role", default="stim_1")
    parser.add_argument(
        "--overview-event-index",
        type=int,
        default=0,
        help="0-based index among metadata-matched anchor events.",
    )

    parser.add_argument("--pre", type=float, default=0.5)
    parser.add_argument("--post", type=float, default=1.0)

    parser.add_argument(
        "--overview-channels",
        nargs="+",
        default=DEFAULT_OVERVIEW_CHANNELS,
        help="Channels shown in the stacked EEG panel.",
    )
    parser.add_argument(
        "--spectrogram-channels",
        nargs="+",
        default=DEFAULT_SPECTROGRAM_CHANNELS,
        help="Channels shown as separate smooth spectrograms on the left.",
    )

    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Use stored stage-3 signal without extra display filtering.",
    )
    parser.add_argument("--lowcut", type=float, default=0.5)
    parser.add_argument("--highcut", type=float, default=40.0)

    parser.add_argument("--spec-fmin", type=float, default=1.0)
    parser.add_argument("--spec-fmax", type=float, default=40.0)
    parser.add_argument(
        "--spec-window",
        type=float,
        default=0.20,
        help="Spectrogram window length in seconds.",
    )
    parser.add_argument(
        "--spec-overlap",
        type=float,
        default=0.95,
        help="Fractional overlap between windows (0 to <1).",
    )
    parser.add_argument(
        "--spec-nfft",
        type=int,
        default=512,
        help="FFT size; larger gives denser frequency grid for a smoother display.",
    )
    parser.add_argument(
        "--power-scale",
        choices=("zscore_per_frequency", "zscore_global", "db"),
        default="zscore_per_frequency",
        help="How to normalize each channel's spectrogram before plotting.",
    )
    parser.add_argument(
        "--smooth-sigma-time",
        type=float,
        default=1.4,
        help="Gaussian smoothing sigma along time bins.",
    )
    parser.add_argument(
        "--smooth-sigma-freq",
        type=float,
        default=1.1,
        help="Gaussian smoothing sigma along frequency bins.",
    )
    parser.add_argument(
        "--contour-percentile",
        type=float,
        default=90.0,
        help="Percentile threshold for dashed white contours. Set <=0 to disable.",
    )
    parser.add_argument(
        "--contour-linewidth",
        type=float,
        default=1.1,
        help="Line width for dashed contour overlays.",
    )
    parser.add_argument(
        "--spec-vmin",
        type=float,
        default=None,
        help="Optional fixed lower color limit.",
    )
    parser.add_argument(
        "--spec-vmax",
        type=float,
        default=None,
        help="Optional fixed upper color limit.",
    )
    parser.add_argument("--cmap", default="jet")

    parser.add_argument("--spacing-factor", type=float, default=3.0)
    parser.add_argument("--line-width", type=float, default=0.8)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def normalize_value(value: Any) -> str:
    text = _to_text(value).strip()
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return " ".join(text.split()).casefold()
    if not math.isfinite(number):
        return text.casefold()
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:g}"


def event_value(event: EventRecord, field_name: str) -> Any:
    if field_name == "type":
        return event.event_type
    if field_name == "event_role":
        return event.event_role
    return event.metadata.get(field_name, "")


def display_value(event: EventRecord, field_name: str) -> str:
    return _to_text(event_value(event, field_name)).strip()


def select_anchor_events(dataset: EEGLABContinuous, args: argparse.Namespace) -> list[EventRecord]:
    requested = {
        "event_role": args.anchor_role,
        "task": args.task,
        "tms": args.tms,
        "tms_target": args.target,
        "category": args.category,
    }

    matches: list[EventRecord] = []
    for event in dataset.events:
        if all(
            normalize_value(event_value(event, field_name))
            == normalize_value(expected)
            for field_name, expected in requested.items()
        ):
            matches.append(event)

    if not matches:
        criteria = ", ".join(f"{key}={value}" for key, value in requested.items())
        raise ValueError(f"No anchor events matched: {criteria}")
    return matches


def apply_display_filter(data_uv: np.ndarray, sfreq: float, args: argparse.Namespace) -> tuple[np.ndarray, str]:
    if args.no_filter:
        return data_uv, "no additional display filter"
    filtered = display_bandpass(data_uv, sfreq, args.lowcut, args.highcut)
    return filtered, f"display filter {args.lowcut:g}-{args.highcut:g} Hz"


def compute_spectrogram_db(
    signal_uv: np.ndarray,
    sfreq: float,
    segment_start_relative: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0 <= args.spec_overlap < 1:
        raise ValueError("--spec-overlap must satisfy 0 <= overlap < 1.")

    nperseg = int(round(args.spec_window * sfreq))
    nperseg = max(8, min(nperseg, signal_uv.size))
    noverlap = int(round(nperseg * args.spec_overlap))
    noverlap = min(noverlap, nperseg - 1)
    nfft = max(int(args.spec_nfft), nperseg)

    freqs, times_from_segment_start, power = spectrogram(
        signal_uv,
        fs=sfreq,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend="constant",
        scaling="density",
        mode="psd",
    )

    relative_times = segment_start_relative + times_from_segment_start
    freq_mask = (freqs >= args.spec_fmin) & (freqs <= args.spec_fmax)
    if not np.any(freq_mask):
        raise ValueError(
            f"No spectrogram frequencies fall inside {args.spec_fmin:g}-{args.spec_fmax:g} Hz."
        )

    freqs = freqs[freq_mask]
    power = power[freq_mask]
    power_db = 10.0 * np.log10(np.maximum(power, np.finfo(float).tiny))
    return freqs, relative_times, power_db


def normalize_power(power_db: np.ndarray, mode: str) -> np.ndarray:
    if mode == "db":
        return power_db
    if mode == "zscore_global":
        mean = power_db.mean()
        std = power_db.std(ddof=0)
        if std <= 0:
            return power_db - mean
        return (power_db - mean) / std

    # z-score each frequency row separately: closest to the provided reference style.
    mean = power_db.mean(axis=1, keepdims=True)
    std = power_db.std(axis=1, keepdims=True, ddof=0)
    std = np.where(std <= 0, 1.0, std)
    return (power_db - mean) / std


def smooth_map(power_map: np.ndarray, sigma_freq: float, sigma_time: float) -> np.ndarray:
    if sigma_freq <= 0 and sigma_time <= 0:
        return power_map
    return gaussian_filter(
        power_map,
        sigma=(max(0.0, sigma_freq), max(0.0, sigma_time)),
        mode="nearest",
    )


def choose_color_limits(spec_maps: list[np.ndarray], args: argparse.Namespace) -> tuple[float, float]:
    combined = np.concatenate([array.ravel() for array in spec_maps])
    combined = combined[np.isfinite(combined)]
    if combined.size == 0:
        return -1.0, 1.0

    auto_vmin = float(np.percentile(combined, 2.0))
    auto_vmax = float(np.percentile(combined, 98.0))
    vmin = auto_vmin if args.spec_vmin is None else args.spec_vmin
    vmax = auto_vmax if args.spec_vmax is None else args.spec_vmax
    if vmin >= vmax:
        raise ValueError("Spectrogram color limits require vmin < vmax.")
    return vmin, vmax


def plot_smooth_spectrogram(
    ax: plt.Axes,
    channel_name: str,
    freqs: np.ndarray,
    times_s: np.ndarray,
    power_map: np.ndarray,
    vmin: float,
    vmax: float,
    args: argparse.Namespace,
):
    extent = [times_s[0] * 1000.0, times_s[-1] * 1000.0, freqs[0], freqs[-1]]
    image = ax.imshow(
        power_map,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=args.cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )
    ax.axvline(0.0, color="k", linestyle=":", linewidth=1.0)
    ax.set_xlim(times_s[0] * 1000.0, times_s[-1] * 1000.0)
    ax.set_ylim(args.spec_fmin, args.spec_fmax)
    ax.set_ylabel("Frequency (Hz)")
    ax.text(
        0.02,
        0.94,
        channel_name,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        fontweight="bold",
        color="white",
        bbox=dict(facecolor="black", edgecolor="none", alpha=0.25, pad=2.0),
    )

    if args.contour_percentile > 0:
        finite = power_map[np.isfinite(power_map)]
        if finite.size > 0:
            level = np.percentile(finite, args.contour_percentile)
            X, Y = np.meshgrid(times_s * 1000.0, freqs)
            ax.contour(
                X,
                Y,
                power_map,
                levels=[level],
                colors="white",
                linewidths=args.contour_linewidth,
                linestyles="--",
                alpha=0.9,
            )
    return image


def plot_stacked_eeg(
    ax: plt.Axes,
    data_uv: np.ndarray,
    relative_times: np.ndarray,
    channel_names: list[str],
    spacing: float,
    anchor: EventRecord,
    filter_text: str,
    args: argparse.Namespace,
) -> None:
    offsets = np.arange(len(channel_names) - 1, -1, -1, dtype=float) * spacing
    x_range = args.pre + args.post

    ax.axvspan(-args.pre, 0.0, color=PRE_BG, alpha=0.75, linewidth=0)
    ax.axvspan(0.0, args.post, color=POST_BG, alpha=0.65, linewidth=0)

    for index, channel_name in enumerate(channel_names):
        ax.plot(relative_times, data_uv[index] + offsets[index], color=EEG_TRACE_COLOR, linewidth=args.line_width)
        ax.text(-args.pre - 0.018 * x_range, offsets[index], channel_name, ha="right", va="center", fontsize=8.5)

    ax.axvline(0.0, color="#b42318", linewidth=1.8)
    ax.text(
        0.0,
        offsets[0] + 0.42 * spacing,
        args.anchor_role,
        rotation=90,
        ha="right",
        va="bottom",
        color="#b42318",
        fontsize=8.5,
    )

    scale_uv = 100.0
    scale_x = args.post - 0.04 * x_range
    scale_bottom = offsets[-1] - 0.70 * spacing
    ax.plot([scale_x, scale_x], [scale_bottom, scale_bottom + scale_uv], color="#111827", linewidth=1.8)
    ax.text(scale_x + 0.012 * x_range, scale_bottom + scale_uv / 2.0, "100 µV", va="center", fontsize=7.5)

    subtitle = (
        f"CSV row {display_value(anchor, 'csv_row') or 'n/a'} | "
        f"trial {display_value(anchor, 'trial_number') or 'n/a'} | "
        f"stimulus {display_value(anchor, 'stimulus') or 'n/a'} | "
        f"phoneme1 {display_value(anchor, 'phoneme1') or 'n/a'} | "
        f"category {display_value(anchor, 'category') or 'n/a'} | "
        f"{filter_text}"
    )
    ax.set_title(f"Continuous EEG overview\n{subtitle}", loc="left", fontsize=9.5)
    ax.set_xlim(-args.pre, args.post)
    ax.set_xlabel(f"Time from {args.anchor_role} (s)")
    ax.set_yticks([])
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.margins(x=0)


def main() -> None:
    args = parse_args()
    if args.pre < 0 or args.post <= 0:
        raise ValueError("--pre must be >= 0 and --post must be > 0.")
    if args.spec_fmin < 0 or args.spec_fmax <= args.spec_fmin:
        raise ValueError("Require 0 <= --spec-fmin < --spec-fmax.")

    dataset = load_eeglab_metadata(args.set)
    matches = select_anchor_events(dataset, args)
    if not 0 <= args.overview_event_index < len(matches):
        raise IndexError(f"--overview-event-index must be in 0..{len(matches)-1}.")

    anchor = matches[args.overview_event_index]
    window_start = max(dataset.xmin, anchor.onset_seconds - args.pre)
    window_stop = min(dataset.xmin + dataset.duration_seconds, anchor.onset_seconds + args.post)

    overview_indices, overview_names = select_channel_indices(dataset.channel_names, args.overview_channels)
    overview_raw, absolute_times, _, _ = read_data_segment(dataset, overview_indices, window_start, window_stop)
    overview_plot, filter_text = apply_display_filter(overview_raw, dataset.sfreq, args)
    relative_times = absolute_times - anchor.onset_seconds

    spec_indices, spec_names = select_channel_indices(dataset.channel_names, args.spectrogram_channels)
    spec_raw, spec_absolute_times, _, _ = read_data_segment(dataset, spec_indices, window_start, window_stop)
    spec_plot, _ = apply_display_filter(spec_raw, dataset.sfreq, args)
    spec_segment_start_relative = spec_absolute_times[0] - anchor.onset_seconds

    spec_results = []
    spec_maps = []
    for channel_signal in spec_plot:
        freqs, spec_times_s, power_db = compute_spectrogram_db(
            channel_signal,
            dataset.sfreq,
            spec_segment_start_relative,
            args,
        )
        power_map = normalize_power(power_db, args.power_scale)
        power_map = smooth_map(power_map, args.smooth_sigma_freq, args.smooth_sigma_time)
        spec_results.append((freqs, spec_times_s, power_map))
        spec_maps.append(power_map)

    vmin, vmax = choose_color_limits(spec_maps, args)
    spacing = robust_spacing(overview_plot, args.spacing_factor)

    n_spec = len(spec_names)
    fig_height = max(7.5, 1.55 * n_spec + 1.2)
    fig = plt.figure(figsize=(15.5, fig_height), constrained_layout=True)
    grid = fig.add_gridspec(
        n_spec,
        2,
        width_ratios=[1.08, 2.15],
        wspace=0.08,
        hspace=0.10,
    )

    spec_axes = []
    image = None
    for index, (channel_name, result) in enumerate(zip(spec_names, spec_results, strict=True)):
        freqs, spec_times_s, power_map = result
        ax = fig.add_subplot(grid[index, 0])
        spec_axes.append(ax)
        image = plot_smooth_spectrogram(ax, channel_name, freqs, spec_times_s, power_map, vmin, vmax, args)
        if index == 0:
            ax.set_title("Smooth spectrogram", loc="left", fontsize=10.0)
        if index < n_spec - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Time (ms)")

    eeg_ax = fig.add_subplot(grid[:, 1])
    plot_stacked_eeg(
        eeg_ax,
        overview_plot,
        relative_times,
        overview_names,
        spacing,
        anchor,
        filter_text,
        args,
    )

    if image is not None:
        cbar = fig.colorbar(image, ax=spec_axes, pad=0.02)
        label = {
            "zscore_per_frequency": "z-scored band power",
            "zscore_global": "global z-scored band power",
            "db": "band power (dB µV²/Hz)",
        }[args.power_scale]
        cbar.set_label(label)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved figure: {output_path}")

    print("\n=== Selection ===")
    print(f"Matched anchors: {len(matches)}")
    print(f"Selected anchor index: {args.overview_event_index}")
    print(f"Anchor onset: {anchor.onset_seconds:.6f} s")
    print(f"Spectrogram channels: {', '.join(spec_names)}")
    print(f"Spectrogram range: {args.spec_fmin:g}-{args.spec_fmax:g} Hz")
    print(f"Power scale: {args.power_scale}")
    print(f"Smoothing sigma (freq, time): ({args.smooth_sigma_freq:g}, {args.smooth_sigma_time:g})")
    print(f"Color limits: {vmin:.3f} to {vmax:.3f}")
    print(f"Display filter: {filter_text}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
