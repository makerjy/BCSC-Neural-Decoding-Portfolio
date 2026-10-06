#!/usr/bin/env python3
"""
Create Figure 1 from a continuous EEGLAB .set/.fdt pair.

The figure is built around stimulus-1 aligned to time zero:
  A. Stacked continuous EEG overview around one selected anchor event.
  B. Same-channel ERP mean with 95% CI across selected trials.
  C. Paired pre/post response metric for each selected trial.

Signal data are read from the .fdt file through NumPy memmap slices. The full
continuous recording is not loaded into RAM.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from plot_stage3_continuous_eeg_2 import (
    EEG_TRACE_COLOR,
    EEGLABContinuous,
    EventRecord,
    _to_text,
    display_bandpass,
    load_eeglab_metadata,
    open_data_matrix,
    read_data_segment,
    robust_spacing,
    select_channel_indices,
)


DEFAULT_OVERVIEW_CHANNELS = [
    "Fp1", "F3", "FC3", "C3", "Cz", "C4",
    "FC4", "P3", "Pz", "P4", "O1", "O2",
]
MOTOR_CHANNELS = ["C3", "Cz", "C4"]
DYNAMIC_CHANNELS = ["Fp1", "F3", "O1"]
DEFAULT_ERP_CHANNELS = MOTOR_CHANNELS

ERP_LINE_COLOR = "#17324d"
ERP_CI_COLOR = "#86a9c6"
STIM_LINE_COLOR = "#b42318"
PRE_REGION_COLOR = "#eef2f6"
POST_REGION_COLOR = "#fff1dc"
PRE_POINT_COLOR = "#6b7280"
POST_POINT_COLOR = "#b45309"

EVENT_ID_FIELDS = ("csv_row", "csv_raw", "trial_number")


@dataclass(frozen=True)
class EpochPlan:
    """Sample bounds for one anchor-centered epoch."""

    event: EventRecord
    start_sample: int
    stop_sample: int
    anchor_sample: int


@dataclass(frozen=True)
class EpochData:
    """Baseline-corrected epochs and selection bookkeeping."""

    epochs_uv: np.ndarray
    times: np.ndarray
    events: list[EventRecord]
    channel_names: list[str]
    raw_match_count: int
    duplicate_count: int
    boundary_excluded_count: int
    event_id_field: str


@dataclass(frozen=True)
class OverviewData:
    """Prepared data for the Figure 1A stacked continuous overview."""

    plot_data_uv: np.ndarray
    plot_times: np.ndarray
    channel_names: list[str]
    window_start: float
    window_stop: float
    anchor: EventRecord
    spacing: float
    filter_text: str


@dataclass(frozen=True)
class ChannelSet:
    """One ERP/pre-post channel set to render."""

    mode: str
    channels: list[str]
    title_note: str = ""


@dataclass(frozen=True)
class ERPClusterStats:
    """One time-adjacent ERP cluster from within-subject trial-level testing."""

    channel: str
    cluster_id: int
    start_time: float
    end_time: float
    cluster_stat: float
    p_corrected: float
    significance: str
    n_trials: int
    start_index: int
    end_index: int


@dataclass(frozen=True)
class PairedChannelStats:
    """Paired pre/post metric statistics for one channel."""

    channel: str
    n: int
    test: str
    pre_mean: float
    post_mean: float
    delta_rms: float
    statistic: float
    p_raw: float
    p_corrected: float
    significance: str


def parse_args() -> argparse.Namespace:
    """Parse command-line options for Figure 1 generation."""

    parser = argparse.ArgumentParser(
        description="Generate Figure 1 from continuous EEGLAB .set/.fdt data."
    )
    parser.add_argument("--set", required=True, help="Path to the EEGLAB .set file.")
    parser.add_argument(
        "--output-dir",
        default="./figures",
        help="Directory where Figure 1 files will be written.",
    )
    parser.add_argument("--show", action="store_true", help="Show figures interactively.")

    parser.add_argument("--task", default="single_phoneme")
    parser.add_argument("--tms", default="0")
    parser.add_argument("--target", default="control", help="Matches event.tms_target.")
    parser.add_argument("--category", default="vowels")
    parser.add_argument("--anchor-role", default="stim_1")

    parser.add_argument("--pre", type=float, default=0.5)
    parser.add_argument("--post", type=float, default=1.0)
    parser.add_argument(
        "--overview-event-index",
        type=int,
        default=0,
        help="0-based index among metadata-matched anchor events for Figure 1A.",
    )
    parser.add_argument(
        "--overview-channels",
        nargs="+",
        default=DEFAULT_OVERVIEW_CHANNELS,
        help="Channels shown in the continuous overview panel.",
    )
    parser.add_argument(
        "--erp-channels",
        nargs="+",
        default=None,
        help=(
            "Custom channels used for both ERP and pre/post panels. "
            "Defaults to C3 Cz C4 when --channel-set is not provided."
        ),
    )
    parser.add_argument(
        "--channel-set",
        choices=("motor", "dynamic", "both"),
        default=None,
        help=(
            "Predefined ERP/pre-post channel set. "
            "motor=C3 Cz C4, dynamic=Fp1 F3 O1, both=renders both sets."
        ),
    )

    parser.add_argument("--baseline-start", type=float, default=-0.5)
    parser.add_argument("--baseline-end", type=float, default=-0.1)
    parser.add_argument(
        "--pre-start",
        type=float,
        default=None,
        help="Start of pre window for quantitative comparison. Defaults to baseline-start.",
    )
    parser.add_argument(
        "--pre-end",
        type=float,
        default=None,
        help="End of pre window for quantitative comparison. Defaults to baseline-end.",
    )
    parser.add_argument("--post-start", type=float, default=0.1)
    parser.add_argument("--post-end", type=float, default=0.5)
    parser.add_argument(
        "--comparison-metric",
        choices=("rms", "mean"),
        default="rms",
        help="Trial-level metric used in Figure 1C.",
    )
    parser.add_argument(
        "--stats-test",
        choices=("wilcoxon", "paired_t"),
        default="wilcoxon",
        help="Paired Pre/Post statistical test for Figure 1C.",
    )
    parser.add_argument(
        "--erp-stats-start",
        type=float,
        default=0.0,
        help="Start of ERP cluster permutation test window in seconds.",
    )
    parser.add_argument(
        "--erp-stats-end",
        type=float,
        default=0.8,
        help="End of ERP cluster permutation test window in seconds.",
    )
    parser.add_argument(
        "--erp-cluster-forming-p",
        type=float,
        default=0.05,
        help="Two-sided cluster-forming p-value threshold for ERP permutation tests.",
    )
    parser.add_argument(
        "--erp-cluster-alpha",
        type=float,
        default=0.05,
        help="Corrected cluster p-value threshold for ERP significance bars.",
    )
    parser.add_argument(
        "--erp-permutations",
        type=int,
        default=5000,
        help="Number of sign-flip permutations for ERP cluster tests.",
    )
    parser.add_argument(
        "--stats-seed",
        type=int,
        default=20240820,
        help="Random seed for reproducible ERP sign-flip permutation tests.",
    )

    parser.add_argument("--spacing-factor", type=float, default=3.0)
    parser.add_argument("--line-width", type=float, default=0.85)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--event-lines",
        action="store_true",
        help="Show non-anchor events inside the Figure 1A overview window.",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Do not apply the additional display filter to Figure 1A.",
    )
    parser.add_argument("--lowcut", type=float, default=0.5)
    parser.add_argument("--highcut", type=float, default=40.0)

    return parser.parse_args()


def set_publication_style() -> None:
    """Apply consistent font sizes and line widths for exported figures."""

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def normalize_value(value: Any) -> str:
    """
    Normalize MATLAB/Python scalar metadata for exact condition comparison.

    Numeric forms such as 0, 0.0, and "0" become the same token. Text is
    stripped and case-folded.
    """

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
    """Read one event metadata field with compatibility for core fields."""

    if field_name == "type":
        return event.event_type
    if field_name == "event_role":
        return event.event_role
    metadata = getattr(event, "metadata", {})
    return metadata.get(field_name, "")


def display_value(event: EventRecord, field_name: str) -> str:
    """Return a display string for one event metadata field."""

    return _to_text(event_value(event, field_name)).strip()


def event_identifier(event: EventRecord) -> tuple[str, str]:
    """
    Return the preferred event identifier using csv_row, csv_raw, then trial_number.
    """

    for field_name in EVENT_ID_FIELDS:
        value = display_value(event, field_name)
        if value:
            return value, field_name
    return f"{event.latency_samples_1based:.3f}", "latency"


def pretty_event_id(event: EventRecord) -> str:
    """Format the preferred event identifier for figure titles and logs."""

    value, field_name = event_identifier(event)
    labels = {
        "csv_row": "CSV row",
        "csv_raw": "CSV raw",
        "trial_number": "Trial number",
        "latency": "Latency sample",
    }
    return f"{labels.get(field_name, field_name)}: {value}"


def infer_subject(dataset: EEGLABContinuous, events: list[EventRecord]) -> str:
    """Infer subject from event metadata, falling back to the .set filename."""

    for event in events:
        subject = display_value(event, "subject")
        if subject:
            return subject
    return dataset.set_path.stem.split("_")[0]


def validate_args(args: argparse.Namespace) -> None:
    """Validate windows and numeric plotting parameters before data access."""

    if args.pre < 0:
        raise ValueError("--pre must be >= 0.")
    if args.post <= 0:
        raise ValueError("--post must be > 0.")
    if args.spacing_factor <= 0:
        raise ValueError("--spacing-factor must be > 0.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be > 0.")
    if args.baseline_start >= args.baseline_end:
        raise ValueError("--baseline-start must be earlier than --baseline-end.")
    if args.erp_stats_start >= args.erp_stats_end:
        raise ValueError("--erp-stats-start must be earlier than --erp-stats-end.")
    if args.erp_permutations < 1:
        raise ValueError("--erp-permutations must be >= 1.")
    for name in ("erp_cluster_forming_p", "erp_cluster_alpha"):
        value = getattr(args, name)
        if not 0 < value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1.")

    tmin = -args.pre
    tmax = args.post
    windows = [
        ("baseline", args.baseline_start, args.baseline_end),
        ("post", args.post_start, args.post_end),
        ("ERP statistics", args.erp_stats_start, args.erp_stats_end),
    ]
    pre_start, pre_end = comparison_pre_window(args)
    windows.append(("pre", pre_start, pre_end))

    for name, start, end in windows:
        if start >= end:
            raise ValueError(f"{name} window start must be earlier than end.")
        if start < tmin or end > tmax:
            raise ValueError(
                f"{name} window {start:g} to {end:g} s is outside the "
                f"epoch window {tmin:g} to {tmax:g} s."
            )


def comparison_pre_window(args: argparse.Namespace) -> tuple[float, float]:
    """Return the pre window used for the quantitative comparison panel."""

    start = args.baseline_start if args.pre_start is None else args.pre_start
    end = args.baseline_end if args.pre_end is None else args.pre_end
    return start, end


def resolve_channel_sets(args: argparse.Namespace) -> list[ChannelSet]:
    """
    Resolve CLI channel options into one or more channel sets to render.

    --erp-channels is treated as a custom single set. --channel-set selects
    predefined motor, dynamic, or both outputs.
    """

    if args.erp_channels is not None and args.channel_set is not None:
        raise ValueError("Use either --erp-channels or --channel-set, not both.")

    if args.erp_channels is not None:
        normalized_channels = [channel.casefold() for channel in args.erp_channels]
        if normalized_channels == [channel.casefold() for channel in MOTOR_CHANNELS]:
            return [ChannelSet(mode="motor", channels=args.erp_channels)]
        if normalized_channels == [channel.casefold() for channel in DYNAMIC_CHANNELS]:
            return [
                ChannelSet(
                    mode="dynamic",
                    channels=args.erp_channels,
                    title_note="spatially distributed / exploratory channels",
                )
            ]
        return [ChannelSet(mode="custom", channels=args.erp_channels)]

    mode = args.channel_set or "motor"
    if mode == "motor":
        return [ChannelSet(mode="motor", channels=MOTOR_CHANNELS)]
    if mode == "dynamic":
        return [
            ChannelSet(
                mode="dynamic",
                channels=DYNAMIC_CHANNELS,
                title_note="spatially distributed / exploratory channels",
            )
        ]
    if mode == "both":
        return [
            ChannelSet(mode="motor", channels=MOTOR_CHANNELS),
            ChannelSet(
                mode="dynamic",
                channels=DYNAMIC_CHANNELS,
                title_note="spatially distributed / exploratory channels",
            ),
        ]

    raise ValueError(f"Unsupported channel set mode: {mode}")


def channel_suffix(channel_names: list[str]) -> str:
    """Return a stable filename suffix for a channel list."""

    cleaned: list[str] = []
    for channel_name in channel_names:
        token = "".join(
            character
            for character in channel_name
            if character.isalnum() or character in {"-", "_"}
        )
        cleaned.append(token or "channel")
    return "_".join(cleaned)


def channel_title(prefix: str, channel_set: ChannelSet) -> str:
    """Return a panel title that optionally labels exploratory channel sets."""

    if channel_set.title_note:
        return f"{prefix} ({channel_set.title_note})"
    if channel_set.mode in {"motor", "dynamic"}:
        return f"{prefix} ({channel_set.mode})"
    return prefix


def select_anchor_events(
    dataset: EEGLABContinuous,
    args: argparse.Namespace,
) -> list[EventRecord]:
    """Select anchor events by exact normalized metadata values."""

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
        criteria = ", ".join(
            f"{field_name}={expected}"
            for field_name, expected in requested.items()
        )
        raise ValueError(f"No anchor events matched requested metadata: {criteria}.")

    return matches


def deduplicate_events(events: list[EventRecord]) -> tuple[list[EventRecord], int, str]:
    """Remove duplicate anchors using the preferred event identifier."""

    seen: set[tuple[str, str]] = set()
    deduplicated: list[EventRecord] = []
    duplicate_count = 0
    event_id_field = "unknown"

    for event in events:
        value, field_name = event_identifier(event)
        if event_id_field == "unknown":
            event_id_field = field_name
        key = (field_name, normalize_value(value))
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        deduplicated.append(event)

    return deduplicated, duplicate_count, event_id_field


def build_epoch_plans(
    dataset: EEGLABContinuous,
    events: list[EventRecord],
    pre: float,
    post: float,
) -> tuple[list[EpochPlan], int, np.ndarray]:
    """
    Convert selected events into sample windows, excluding boundary conflicts.
    """

    pre_samples = int(round(pre * dataset.sfreq))
    post_samples = int(round(post * dataset.sfreq))
    times = np.arange(-pre_samples, post_samples + 1, dtype=np.float64) / dataset.sfreq

    plans: list[EpochPlan] = []
    boundary_excluded_count = 0
    for event in events:
        anchor_sample = int(round(event.latency_samples_1based - 1.0))
        start_sample = anchor_sample - pre_samples
        stop_sample = anchor_sample + post_samples + 1

        if start_sample < 0 or stop_sample > dataset.n_samples:
            boundary_excluded_count += 1
            continue

        plans.append(
            EpochPlan(
                event=event,
                start_sample=start_sample,
                stop_sample=stop_sample,
                anchor_sample=anchor_sample,
            )
        )

    return plans, boundary_excluded_count, times


def window_mask(times: np.ndarray, start: float, end: float) -> np.ndarray:
    """Return a boolean mask for a closed time window."""

    mask = (times >= start) & (times <= end)
    if not np.any(mask):
        raise ValueError(f"Window {start:g} to {end:g} s contains no samples.")
    return mask


def check_finite(label: str, data: np.ndarray) -> int:
    """Count non-finite values and fail if any are present."""

    count = int(np.size(data) - np.count_nonzero(np.isfinite(data)))
    print(f"{label} non-finite samples: {count}")
    if count:
        raise ValueError(f"{label} contains {count} NaN/Inf samples.")
    return count


def load_epochs(
    dataset: EEGLABContinuous,
    plans: list[EpochPlan],
    channel_indices: list[int],
) -> np.ndarray:
    """
    Read anchor-centered epochs from the continuous data matrix.

    The returned array has shape trials x channels x samples.
    """

    if not plans:
        raise ValueError("No valid epoch windows are available.")

    data = open_data_matrix(dataset)
    n_trials = len(plans)
    n_channels = len(channel_indices)
    n_times = plans[0].stop_sample - plans[0].start_sample
    epochs = np.empty((n_trials, n_channels, n_times), dtype=np.float64)
    channel_index_array = np.asarray(channel_indices)

    for trial_index, plan in enumerate(plans):
        epochs[trial_index] = np.asarray(
            data[channel_index_array, plan.start_sample:plan.stop_sample],
            dtype=np.float64,
        )

    return epochs


def baseline_correct(
    epochs: np.ndarray,
    times: np.ndarray,
    baseline_start: float,
    baseline_end: float,
) -> np.ndarray:
    """Apply per-trial, per-channel baseline correction."""

    mask = window_mask(times, baseline_start, baseline_end)
    baseline = epochs[:, :, mask].mean(axis=2, keepdims=True)
    return epochs - baseline


def prepare_epoch_data(
    dataset: EEGLABContinuous,
    args: argparse.Namespace,
    requested_channels: list[str],
    raw_matches: list[EventRecord] | None = None,
) -> EpochData:
    """Select events, read epochs, and apply baseline correction."""

    if raw_matches is None:
        raw_matches = select_anchor_events(dataset, args)
    deduplicated, duplicate_count, event_id_field = deduplicate_events(raw_matches)
    plans, boundary_excluded_count, times = build_epoch_plans(
        dataset=dataset,
        events=deduplicated,
        pre=args.pre,
        post=args.post,
    )

    if not plans:
        raise ValueError(
            "No valid epochs remain after boundary exclusion "
            f"({boundary_excluded_count} excluded)."
        )

    channel_indices, selected_names = select_channel_indices(
        dataset.channel_names,
        requested_channels,
    )
    epochs = load_epochs(dataset, plans, channel_indices)
    check_finite("Epoch data before baseline correction", epochs)
    epochs = baseline_correct(
        epochs,
        times,
        args.baseline_start,
        args.baseline_end,
    )
    check_finite("Epoch data after baseline correction", epochs)

    return EpochData(
        epochs_uv=epochs,
        times=times,
        events=[plan.event for plan in plans],
        channel_names=selected_names,
        raw_match_count=len(raw_matches),
        duplicate_count=duplicate_count,
        boundary_excluded_count=boundary_excluded_count,
        event_id_field=event_id_field,
    )


def prepare_overview_data(
    dataset: EEGLABContinuous,
    args: argparse.Namespace,
    matched_events: list[EventRecord],
) -> OverviewData:
    """Read and optionally display-filter the continuous overview segment."""

    if not 0 <= args.overview_event_index < len(matched_events):
        raise IndexError(
            "--overview-event-index "
            f"{args.overview_event_index} is outside 0..{len(matched_events) - 1}."
        )

    anchor = matched_events[args.overview_event_index]
    window_start = max(dataset.xmin, anchor.onset_seconds - args.pre)
    window_stop = min(
        dataset.xmin + dataset.duration_seconds,
        anchor.onset_seconds + args.post,
    )
    channel_indices, selected_names = select_channel_indices(
        dataset.channel_names,
        args.overview_channels,
    )
    data_uv, absolute_times, _, _ = read_data_segment(
        dataset,
        channel_indices,
        window_start,
        window_stop,
    )
    check_finite("Overview data before display filter", data_uv)

    if args.no_filter:
        plot_data_uv = data_uv
        filter_text = "no additional display filter"
    else:
        plot_data_uv = display_bandpass(
            data_uv,
            dataset.sfreq,
            args.lowcut,
            args.highcut,
        )
        filter_text = f"display filter {args.lowcut:g}-{args.highcut:g} Hz"

    check_finite("Overview plotted data", plot_data_uv)
    spacing = robust_spacing(plot_data_uv, args.spacing_factor)

    return OverviewData(
        plot_data_uv=plot_data_uv,
        plot_times=absolute_times - anchor.onset_seconds,
        channel_names=selected_names,
        window_start=window_start,
        window_stop=window_stop,
        anchor=anchor,
        spacing=spacing,
        filter_text=filter_text,
    )


def erp_mean_ci(epochs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean ERP and 95% confidence interval across trials."""

    n_trials = epochs.shape[0]
    mean = epochs.mean(axis=0)
    if n_trials < 2:
        ci = np.zeros_like(mean)
    else:
        sem = epochs.std(axis=0, ddof=1) / math.sqrt(n_trials)
        ci = sem * stats.t.ppf(0.975, n_trials - 1)
    return mean, mean - ci, mean + ci


def compute_metric(
    epochs: np.ndarray,
    times: np.ndarray,
    start: float,
    end: float,
    metric: str,
) -> np.ndarray:
    """Compute trial x channel window metrics for Figure 1C."""

    mask = window_mask(times, start, end)
    window = epochs[:, :, mask]
    if metric == "rms":
        return np.sqrt(np.mean(np.square(window), axis=2))
    if metric == "mean":
        return np.mean(window, axis=2)
    raise ValueError(f"Unsupported metric: {metric}")


def p_to_stars(p_value: float, alpha: float = 0.05) -> str:
    """Convert a p-value into reference-style significance labels."""

    if not np.isfinite(p_value) or p_value >= alpha:
        return "n.s."
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    return "*"


def format_p_value(p_value: float, prefix: str = "p") -> str:
    """Format exact p-values compactly for figures and console output."""

    if not np.isfinite(p_value):
        return f"{prefix}=NA"
    if p_value < 0.001:
        return f"{prefix}<0.001"
    return f"{prefix}={p_value:.3f}"


def find_significant_clusters(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous True regions as inclusive start/end index pairs."""

    if mask.ndim != 1:
        raise ValueError("Cluster mask must be one-dimensional.")
    if not np.any(mask):
        return []

    padded = np.concatenate(([False], mask, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    stops = np.flatnonzero(padded[:-1] & ~padded[1:]) - 1
    return [(int(start), int(stop)) for start, stop in zip(starts, stops, strict=True)]


def one_sample_t_values(data: np.ndarray) -> np.ndarray:
    """Compute one-sample t-values against zero along the trial axis."""

    n_trials = data.shape[0]
    if n_trials < 2:
        return np.zeros(data.shape[1], dtype=np.float64)

    mean = data.mean(axis=0)
    std = data.std(axis=0, ddof=1)
    sem = std / math.sqrt(n_trials)
    return np.divide(
        mean,
        sem,
        out=np.zeros_like(mean, dtype=np.float64),
        where=sem > 0,
    )


def max_cluster_statistic(t_values: np.ndarray, threshold: float) -> float:
    """Return the largest two-sided cluster mass in one t-value vector."""

    clusters = find_significant_clusters(np.abs(t_values) > threshold)
    if not clusters:
        return 0.0
    return float(max(np.sum(np.abs(t_values[start : stop + 1])) for start, stop in clusters))


def permutation_max_cluster_stats(
    data: np.ndarray,
    threshold: float,
    n_permutations: int,
    rng: np.random.Generator,
    chunk_size: int = 512,
) -> np.ndarray:
    """
    Build a sign-flip max-cluster null distribution.

    This is within-subject trial-level exploratory testing: trials are repeated
    observations from one subject, not independent subjects.
    """

    n_trials = data.shape[0]
    sum_squares = np.sum(np.square(data), axis=0)
    max_stats = np.zeros(n_permutations, dtype=np.float64)
    completed = 0

    while completed < n_permutations:
        this_chunk = min(chunk_size, n_permutations - completed)
        signs = rng.choice(
            np.array([-1.0, 1.0], dtype=np.float64),
            size=(this_chunk, n_trials),
        )
        means = (signs @ data) / n_trials
        variances = (sum_squares[np.newaxis, :] - n_trials * np.square(means)) / (n_trials - 1)
        variances = np.maximum(variances, 0.0)
        sems = np.sqrt(variances / n_trials)
        t_values = np.divide(
            means,
            sems,
            out=np.zeros_like(means, dtype=np.float64),
            where=sems > 0,
        )

        for row_index in range(this_chunk):
            max_stats[completed + row_index] = max_cluster_statistic(
                t_values[row_index],
                threshold,
            )

        completed += this_chunk

    return max_stats


def cluster_permutation_1samp(
    data: np.ndarray,
    times: np.ndarray,
    channel: str,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> list[ERPClusterStats]:
    """
    Run a two-sided sign-flip one-sample cluster permutation test for one channel.

    The input is trial x time baseline-corrected ERP data for a single subject.
    Resulting p-values are cluster-level corrected across the tested time points.
    """

    test_mask = window_mask(times, args.erp_stats_start, args.erp_stats_end)
    full_indices = np.flatnonzero(test_mask)
    data_window = data[:, test_mask]
    finite_trials = np.all(np.isfinite(data_window), axis=1)
    data_window = data_window[finite_trials]
    n_trials = data_window.shape[0]

    if n_trials < 2:
        return []

    threshold = float(
        stats.t.ppf(1.0 - args.erp_cluster_forming_p / 2.0, df=n_trials - 1)
    )
    observed_t = one_sample_t_values(data_window)
    clusters = find_significant_clusters(np.abs(observed_t) > threshold)
    if not clusters:
        return []

    null_max_stats = permutation_max_cluster_stats(
        data_window,
        threshold,
        args.erp_permutations,
        rng,
    )

    cluster_results: list[ERPClusterStats] = []
    for cluster_id, (start, stop) in enumerate(clusters, start=1):
        cluster_stat = float(np.sum(np.abs(observed_t[start : stop + 1])))
        p_corrected = float(
            (np.count_nonzero(null_max_stats >= cluster_stat) + 1)
            / (args.erp_permutations + 1)
        )
        start_index = int(full_indices[start])
        end_index = int(full_indices[stop])
        cluster_results.append(
            ERPClusterStats(
                channel=channel,
                cluster_id=cluster_id,
                start_time=float(times[start_index]),
                end_time=float(times[end_index]),
                cluster_stat=cluster_stat,
                p_corrected=p_corrected,
                significance=p_to_stars(p_corrected, args.erp_cluster_alpha),
                n_trials=n_trials,
                start_index=start_index,
                end_index=end_index,
            )
        )

    return cluster_results


def compute_erp_cluster_statistics(
    epoch_data: EpochData,
    args: argparse.Namespace,
) -> dict[str, list[ERPClusterStats]]:
    """Compute ERP cluster statistics for every plotted channel."""

    rng = np.random.default_rng(args.stats_seed)
    results: dict[str, list[ERPClusterStats]] = {}
    for channel_index, channel in enumerate(epoch_data.channel_names):
        results[channel] = cluster_permutation_1samp(
            epoch_data.epochs_uv[:, channel_index, :],
            epoch_data.times,
            channel,
            args,
            rng,
        )
    return results


def compute_pre_post_values(
    epoch_data: EpochData,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute trial x channel Pre and Post metrics for Figure 1C."""

    pre_start, pre_end = comparison_pre_window(args)
    pre_values = compute_metric(
        epoch_data.epochs_uv,
        epoch_data.times,
        pre_start,
        pre_end,
        args.comparison_metric,
    )
    post_values = compute_metric(
        epoch_data.epochs_uv,
        epoch_data.times,
        args.post_start,
        args.post_end,
        args.comparison_metric,
    )
    return pre_values, post_values


def holm_correction(p_values: list[float]) -> list[float]:
    """Apply Holm correction to a family of channel-wise p-values."""

    if not p_values:
        return []

    p_array = np.asarray(
        [p if np.isfinite(p) else 1.0 for p in p_values],
        dtype=np.float64,
    )
    order = np.argsort(p_array)
    adjusted_sorted = np.empty_like(p_array)
    running_max = 0.0
    n_tests = len(p_array)

    for rank, original_index in enumerate(order):
        adjusted = min(1.0, (n_tests - rank) * p_array[original_index])
        running_max = max(running_max, adjusted)
        adjusted_sorted[rank] = running_max

    adjusted = np.empty_like(p_array)
    adjusted[order] = adjusted_sorted
    return [float(value) for value in adjusted]


def compute_paired_statistics(
    pre_values: np.ndarray,
    post_values: np.ndarray,
    channel_names: list[str],
    args: argparse.Namespace,
) -> list[PairedChannelStats]:
    """
    Compute paired Pre/Post statistics and Holm-correct p-values across channels.

    These are within-subject trial-level exploratory statistics for one subject,
    not population-level inference.
    """

    raw_results: list[PairedChannelStats] = []
    raw_p_values: list[float] = []

    for channel_index, channel in enumerate(channel_names):
        pre = pre_values[:, channel_index]
        post = post_values[:, channel_index]
        finite_mask = np.isfinite(pre) & np.isfinite(post)
        pre = pre[finite_mask]
        post = post[finite_mask]
        n = int(pre.size)

        if n < 2:
            statistic = math.nan
            p_raw = math.nan
        elif args.stats_test == "paired_t":
            test_result = stats.ttest_rel(post, pre)
            statistic = float(test_result.statistic)
            p_raw = float(test_result.pvalue)
        else:
            differences = post - pre
            if np.allclose(differences, 0.0):
                statistic = 0.0
                p_raw = 1.0
            else:
                test_result = stats.wilcoxon(
                    post,
                    pre,
                    zero_method="pratt",
                    alternative="two-sided",
                    method="auto",
                )
                statistic = float(test_result.statistic)
                p_raw = float(test_result.pvalue)

        pre_mean = float(np.mean(pre)) if n else math.nan
        post_mean = float(np.mean(post)) if n else math.nan
        raw_p_values.append(p_raw)
        raw_results.append(
            PairedChannelStats(
                channel=channel,
                n=n,
                test=args.stats_test,
                pre_mean=pre_mean,
                post_mean=post_mean,
                delta_rms=post_mean - pre_mean,
                statistic=statistic,
                p_raw=p_raw,
                p_corrected=math.nan,
                significance="n.s.",
            )
        )

    corrected_values = holm_correction(raw_p_values)
    corrected_results: list[PairedChannelStats] = []
    for result, p_corrected in zip(raw_results, corrected_values, strict=True):
        corrected_results.append(
            PairedChannelStats(
                channel=result.channel,
                n=result.n,
                test=result.test,
                pre_mean=result.pre_mean,
                post_mean=result.post_mean,
                delta_rms=result.delta_rms,
                statistic=result.statistic,
                p_raw=result.p_raw,
                p_corrected=p_corrected,
                significance=p_to_stars(p_corrected),
            )
        )

    return corrected_results


def shared_erp_ylim(lower: np.ndarray, upper: np.ndarray) -> tuple[float, float]:
    """Return one symmetric y-limit for all ERP channel panels."""

    max_abs = float(np.nanmax(np.abs(np.concatenate([lower.ravel(), upper.ravel()]))))
    if not np.isfinite(max_abs) or max_abs <= 0:
        max_abs = 1.0
    return -1.12 * max_abs, 1.12 * max_abs


def shared_metric_ylim(pre_values: np.ndarray, post_values: np.ndarray) -> tuple[float, float]:
    """Return one y-limit for all quantitative pre/post panels."""

    values = np.concatenate([pre_values.ravel(), post_values.ravel()])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -1.0, 1.0

    low = float(values.min())
    high = float(values.max())
    if abs(high - low) < 1e-12:
        pad = max(abs(high) * 0.1, 1.0)
        return low - pad, high + pad

    pad = 0.35 * (high - low)
    if low >= 0:
        return 0.0, high + pad
    return low - pad, high + pad


def add_panel_label(ax: plt.Axes, label: str) -> None:
    """Add a bold panel label anchored to the upper-left of an axes."""

    ax.text(
        -0.045,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )


def add_figure_stats_legend(fig: plt.Figure, args: argparse.Namespace, include_erp: bool = True) -> None:
    """Add one compact statistical-method note at the bottom of a figure."""

    rms_test = "paired Wilcoxon" if args.stats_test == "wilcoxon" else "paired t-test"
    parts: list[str] = []
    if include_erp:
        parts.append(
            f"ERP: cluster-based sign-flip permutation test "
            f"({args.erp_permutations:,} permutations, corrected cluster p<{args.erp_cluster_alpha:g})"
        )
    parts.append(f"RMS: {rms_test}, Holm-corrected across channels")
    parts.append("* p<0.05, ** p<0.01, *** p<0.001, n.s. p>=0.05")
    fig.text(
        0.5,
        -0.025,
        " | ".join(parts),
        ha="center",
        va="top",
        fontsize=7.3,
        color="#374151",
    )


def add_cluster_bars(
    ax: plt.Axes,
    clusters: list[ERPClusterStats],
    args: argparse.Namespace,
    base_ylim: tuple[float, float],
) -> None:
    """Draw reference-style ERP significance bars for significant clusters."""

    significant_clusters = [
        cluster
        for cluster in clusters
        if np.isfinite(cluster.p_corrected)
        and cluster.p_corrected < args.erp_cluster_alpha
    ]
    y_min, y_max = base_ylim
    span = y_max - y_min
    bar_y = y_max + 0.065 * span

    if not significant_clusters:
        ax.text(
            0.98,
            0.91,
            "n.s.",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            color="#111827",
        )
        return

    for cluster in significant_clusters:
        ax.plot(
            [cluster.start_time, cluster.end_time],
            [bar_y, bar_y],
            color="#111827",
            linewidth=2.6,
            solid_capstyle="butt",
            zorder=6,
        )
        ax.text(
            (cluster.start_time + cluster.end_time) / 2.0,
            bar_y + 0.025 * span,
            cluster.significance,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#111827",
            zorder=7,
        )


def add_significance_bracket(
    ax: plt.Axes,
    paired_stat: PairedChannelStats | None,
    data_max: float,
) -> None:
    """Add a Pre/Post significance bracket and corrected p-value annotation."""

    y_min, y_max = ax.get_ylim()
    span = y_max - y_min
    bracket_y = data_max + 0.075 * span
    bracket_y = min(bracket_y, y_max - 0.24 * span)
    bracket_height = 0.04 * span

    ax.plot(
        [0, 0, 1, 1],
        [bracket_y, bracket_y + bracket_height, bracket_y + bracket_height, bracket_y],
        color="#111827",
        linewidth=0.9,
        clip_on=False,
        zorder=5,
    )

    if paired_stat is None:
        label = "n.s.\npHolm=NA"
    else:
        label = (
            f"{paired_stat.significance}\n"
            f"{format_p_value(paired_stat.p_corrected, prefix='pHolm')}"
        )
    ax.text(
        0.5,
        bracket_y + bracket_height + 0.018 * span,
        label,
        ha="center",
        va="bottom",
        fontsize=8.0,
        color="#111827",
    )


def plot_overview_panel(
    ax: plt.Axes,
    dataset: EEGLABContinuous,
    overview: OverviewData,
    args: argparse.Namespace,
    title_prefix: str = "A. Continuous EEG overview",
) -> None:
    """Draw the stacked continuous EEG panel."""

    offsets = (
        np.arange(len(overview.channel_names) - 1, -1, -1, dtype=np.float64)
        * overview.spacing
    )
    x_range = args.pre + args.post

    ax.axvspan(-args.pre, 0.0, color=PRE_REGION_COLOR, alpha=0.75, linewidth=0)
    ax.axvspan(0.0, args.post, color=POST_REGION_COLOR, alpha=0.65, linewidth=0)

    for index, channel_name in enumerate(overview.channel_names):
        ax.plot(
            overview.plot_times,
            overview.plot_data_uv[index] + offsets[index],
            color=EEG_TRACE_COLOR,
            linewidth=args.line_width,
        )
        ax.text(
            -args.pre - 0.025 * x_range,
            offsets[index],
            channel_name,
            ha="right",
            va="center",
            fontsize=8.5,
        )

    ax.axvline(0.0, color=STIM_LINE_COLOR, linestyle="-", linewidth=1.8)
    ax.text(
        0.0,
        offsets[0] + 0.45 * overview.spacing,
        "stim_1",
        rotation=90,
        ha="right",
        va="bottom",
        color=STIM_LINE_COLOR,
        fontsize=8.5,
    )

    if args.event_lines:
        for event in dataset.events:
            if not overview.window_start <= event.onset_seconds <= overview.window_stop:
                continue
            x_position = event.onset_seconds - overview.anchor.onset_seconds
            if abs(x_position) < 0.25 / dataset.sfreq:
                continue
            ax.axvline(
                x_position,
                color="#8a8f98",
                linestyle=":",
                linewidth=0.6,
                alpha=0.45,
            )
            role = display_value(event, "event_role")
            if role:
                ax.text(
                    x_position,
                    offsets[0] + 0.25 * overview.spacing,
                    role,
                    rotation=90,
                    ha="center",
                    va="bottom",
                    color="#6b7280",
                    fontsize=6.5,
                )

    scale_uv = 100.0
    scale_x = args.post - 0.04 * x_range
    scale_bottom = offsets[-1] - 0.72 * overview.spacing
    ax.plot(
        [scale_x, scale_x],
        [scale_bottom, scale_bottom + scale_uv],
        color="#111827",
        linewidth=1.8,
    )
    ax.text(
        scale_x + 0.012 * x_range,
        scale_bottom + scale_uv / 2,
        "100 uV",
        va="center",
        fontsize=7.5,
    )

    subtitle = (
        f"{pretty_event_id(overview.anchor)} | trial "
        f"{display_value(overview.anchor, 'trial_number') or 'n/a'} | "
        f"stimulus {display_value(overview.anchor, 'stimulus') or 'n/a'} | "
        f"phoneme1 {display_value(overview.anchor, 'phoneme1') or 'n/a'} | "
        f"category {display_value(overview.anchor, 'category') or 'n/a'} | "
        f"{overview.filter_text}"
    )
    ax.set_title(f"{title_prefix}\n{subtitle}", loc="left", fontsize=9.5)
    ax.set_xlim(-args.pre, args.post)
    ax.set_xlabel("Time from stim_1 (s)")
    ax.set_yticks([])
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.margins(x=0.0)


def plot_erp_panels(
    axes: list[plt.Axes] | np.ndarray,
    epoch_data: EpochData,
    args: argparse.Namespace,
    title_prefix: str = "B. ERP",
    erp_cluster_stats: dict[str, list[ERPClusterStats]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    """Draw channel-wise mean ERP with 95% CI."""

    axes_list = list(np.ravel(axes))
    mean, lower, upper = erp_mean_ci(epoch_data.epochs_uv)
    base_y_lim = shared_erp_ylim(lower, upper)
    y_span = base_y_lim[1] - base_y_lim[0]
    y_lim = (base_y_lim[0], base_y_lim[1] + 0.18 * y_span)

    for index, (ax, channel_name) in enumerate(
        zip(axes_list, epoch_data.channel_names, strict=True)
    ):
        ax.axvspan(-args.pre, 0.0, color=PRE_REGION_COLOR, alpha=0.55, linewidth=0)
        ax.axvspan(0.0, args.post, color=POST_REGION_COLOR, alpha=0.45, linewidth=0)
        ax.fill_between(
            epoch_data.times,
            lower[index],
            upper[index],
            color=ERP_CI_COLOR,
            alpha=0.42,
            linewidth=0,
            label="95% CI" if index == 0 else None,
        )
        ax.plot(
            epoch_data.times,
            mean[index],
            color=ERP_LINE_COLOR,
            linewidth=1.45,
            label="Mean ERP" if index == 0 else None,
        )
        ax.axvline(0.0, color=STIM_LINE_COLOR, linewidth=1.2)
        ax.axhline(0.0, color="#9ca3af", linewidth=0.6, alpha=0.65)
        ax.set_xlim(-args.pre, args.post)
        ax.set_ylim(*y_lim)
        if erp_cluster_stats is not None:
            add_cluster_bars(
                ax,
                erp_cluster_stats.get(channel_name, []),
                args,
                base_y_lim,
            )
        ax.set_ylabel(f"{channel_name}\nuV")
        ax.spines[["right", "top"]].set_visible(False)
        if index == 0:
            ax.set_title(
                f"{title_prefix} (mean +/- 95% CI, n={len(epoch_data.events)})",
                loc="left",
                fontsize=9.5,
            )
            ax.legend(frameon=False, loc="upper right")
        if index < len(axes_list) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Time from stim_1 (s)")

    return mean, lower, upper, y_lim


def plot_pre_post_panels(
    axes: list[plt.Axes] | np.ndarray,
    epoch_data: EpochData,
    args: argparse.Namespace,
    title_prefix: str = "C. Pre vs Post",
    pre_values: np.ndarray | None = None,
    post_values: np.ndarray | None = None,
    paired_stats: list[PairedChannelStats] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Draw paired pre/post quantitative comparison panels."""

    axes_list = list(np.ravel(axes))
    pre_start, pre_end = comparison_pre_window(args)
    if pre_values is None or post_values is None:
        pre_values, post_values = compute_pre_post_values(epoch_data, args)
    y_lim = shared_metric_ylim(pre_values, post_values)
    rng = np.random.default_rng(202401)
    stats_by_channel = {
        paired_stat.channel: paired_stat
        for paired_stat in paired_stats or []
    }

    for channel_index, (ax, channel_name) in enumerate(
        zip(axes_list, epoch_data.channel_names, strict=True)
    ):
        x_pre = np.zeros(pre_values.shape[0])
        x_post = np.ones(post_values.shape[0])
        jitter = rng.uniform(-0.045, 0.045, size=pre_values.shape[0])

        for trial_index in range(pre_values.shape[0]):
            ax.plot(
                [x_pre[trial_index] + jitter[trial_index], x_post[trial_index] + jitter[trial_index]],
                [pre_values[trial_index, channel_index], post_values[trial_index, channel_index]],
                color="#9ca3af",
                alpha=0.16,
                linewidth=0.55,
                zorder=1,
            )

        ax.scatter(
            x_pre + jitter,
            pre_values[:, channel_index],
            s=12,
            color=PRE_POINT_COLOR,
            alpha=0.55,
            linewidths=0,
            zorder=2,
            label="Pre" if channel_index == 0 else None,
        )
        ax.scatter(
            x_post + jitter,
            post_values[:, channel_index],
            s=12,
            color=POST_POINT_COLOR,
            alpha=0.65,
            linewidths=0,
            zorder=2,
            label="Post" if channel_index == 0 else None,
        )

        means = [
            float(pre_values[:, channel_index].mean()),
            float(post_values[:, channel_index].mean()),
        ]
        paired_stat = stats_by_channel.get(channel_name)
        delta = (
            paired_stat.delta_rms
            if paired_stat is not None
            else float((post_values[:, channel_index] - pre_values[:, channel_index]).mean())
        )
        ax.plot(
            [0, 1],
            means,
            color="#111827",
            linewidth=2.2,
            marker="o",
            markersize=4.8,
            zorder=3,
        )
        delta_label = "\u0394RMS" if args.comparison_metric == "rms" else "\u0394Mean"
        ax.text(
            0.98,
            0.92,
            f"{delta_label} = {delta:+.1f} \u00b5V",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            color="#111827",
        )

        ax.set_xlim(-0.35, 1.35)
        ax.set_ylim(*y_lim)
        channel_data_max = float(
            np.nanmax(
                np.concatenate(
                    [
                        pre_values[:, channel_index],
                        post_values[:, channel_index],
                    ]
                )
            )
        )
        add_significance_bracket(ax, paired_stat, channel_data_max)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            [
                f"Pre\n{pre_start:g} to {pre_end:g}s",
                f"Post\n{args.post_start:g} to {args.post_end:g}s",
            ]
        )
        metric_label = "RMS amplitude" if args.comparison_metric == "rms" else "Mean amplitude"
        ax.set_ylabel(f"{channel_name}\n{metric_label} (uV)")
        ax.spines[["right", "top"]].set_visible(False)
        if channel_index == 0:
            ax.set_title(f"{title_prefix} ({args.comparison_metric.upper()})", loc="left", fontsize=9.5)

    return pre_values, post_values, y_lim


def save_figure(fig: plt.Figure, output_base: Path, dpi: int) -> None:
    """Save one figure as PNG and PDF."""

    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")


def csv_float(value: float, digits: int = 10) -> str:
    """Format numeric CSV values without rounding very small p-values to zero."""

    if not np.isfinite(value):
        return ""
    return f"{value:.{digits}g}"


def save_erp_cluster_stats_csv(
    output_path: Path,
    epoch_data: EpochData,
    erp_cluster_stats: dict[str, list[ERPClusterStats]],
) -> None:
    """Save ERP cluster permutation statistics to CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "channel",
        "cluster_id",
        "start_time",
        "end_time",
        "cluster_stat",
        "p_corrected",
        "significance",
        "n_trials",
    ]
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for channel in epoch_data.channel_names:
            clusters = erp_cluster_stats.get(channel, [])
            if not clusters:
                writer.writerow(
                    {
                        "channel": channel,
                        "cluster_id": "",
                        "start_time": "",
                        "end_time": "",
                        "cluster_stat": "",
                        "p_corrected": "",
                        "significance": "n.s.",
                        "n_trials": len(epoch_data.events),
                    }
                )
                continue

            for cluster in clusters:
                writer.writerow(
                    {
                        "channel": cluster.channel,
                        "cluster_id": cluster.cluster_id,
                        "start_time": csv_float(cluster.start_time, 8),
                        "end_time": csv_float(cluster.end_time, 8),
                        "cluster_stat": csv_float(cluster.cluster_stat, 10),
                        "p_corrected": csv_float(cluster.p_corrected, 10),
                        "significance": cluster.significance,
                        "n_trials": cluster.n_trials,
                    }
                )
    print(f"Saved stats: {output_path}")


def save_pre_post_stats_csv(
    output_path: Path,
    pre_post_stats: list[PairedChannelStats],
) -> None:
    """Save paired Pre/Post statistics to CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "channel",
        "n",
        "test",
        "pre_mean",
        "post_mean",
        "delta_rms",
        "statistic",
        "p_raw",
        "p_corrected",
        "significance",
    ]
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in pre_post_stats:
            writer.writerow(
                {
                    "channel": result.channel,
                    "n": result.n,
                    "test": result.test,
                    "pre_mean": csv_float(result.pre_mean, 10),
                    "post_mean": csv_float(result.post_mean, 10),
                    "delta_rms": csv_float(result.delta_rms, 10),
                    "statistic": csv_float(result.statistic, 10),
                    "p_raw": csv_float(result.p_raw, 10),
                    "p_corrected": csv_float(result.p_corrected, 10),
                    "significance": result.significance,
                }
            )
    print(f"Saved stats: {output_path}")


def print_erp_cluster_statistics(
    erp_cluster_stats: dict[str, list[ERPClusterStats]],
    channel_names: list[str],
    args: argparse.Namespace,
) -> None:
    """Print significant ERP cluster statistics to console."""

    print("\nERP cluster statistics")
    print(
        "Within-subject trial-level exploratory statistics for one subject; "
        "not population-level inference."
    )
    for channel in channel_names:
        print(f"\n{channel}")
        significant_clusters = [
            cluster
            for cluster in erp_cluster_stats.get(channel, [])
            if np.isfinite(cluster.p_corrected)
            and cluster.p_corrected < args.erp_cluster_alpha
        ]
        if not significant_clusters:
            print("No significant cluster")
            continue

        for cluster_index, cluster in enumerate(significant_clusters, start=1):
            print(
                f"Cluster {cluster_index}: "
                f"{cluster.start_time:.3f}-{cluster.end_time:.3f} s, "
                f"{format_p_value(cluster.p_corrected)}, "
                f"{cluster.significance}"
            )


def print_pre_post_statistics(pre_post_stats: list[PairedChannelStats]) -> None:
    """Print paired Pre/Post statistics to console."""

    print("\nPre/Post metric statistics")
    print(
        "Within-subject trial-level exploratory statistics for one subject; "
        "not population-level inference."
    )
    for result in pre_post_stats:
        print(
            f"{result.channel}: n={result.n}, {result.test}, "
            f"delta={result.delta_rms:+.3f} uV, "
            f"raw {format_p_value(result.p_raw)}, "
            f"Holm {format_p_value(result.p_corrected)}, "
            f"{result.significance}"
        )


def create_overview_figure(
    dataset: EEGLABContinuous,
    overview: OverviewData,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    """Create and save the shared Figure 1A overview panel."""
    overview_height = max(5.1, 0.37 * len(overview.channel_names) + 1.5)
    fig_a, ax_a = plt.subplots(figsize=(12.5, overview_height), constrained_layout=True)
    plot_overview_panel(ax_a, dataset, overview, args)
    save_figure(fig_a, output_dir / "figure1_overview", args.dpi)
    plt.close(fig_a)


def create_channel_figures(
    epoch_data: EpochData,
    channel_set: ChannelSet,
    args: argparse.Namespace,
    output_dir: Path,
    save_legacy_names: bool,
    erp_cluster_stats: dict[str, list[ERPClusterStats]],
    pre_values: np.ndarray,
    post_values: np.ndarray,
    pre_post_stats: list[PairedChannelStats],
) -> tuple[tuple[float, float], tuple[float, float], np.ndarray, np.ndarray]:
    """Create and save ERP and pre/post panels for one channel set."""

    suffix = channel_suffix(epoch_data.channel_names)
    fig_b, axes_b = plt.subplots(
        len(epoch_data.channel_names),
        1,
        figsize=(7.2, 6.5),
        sharex=True,
        constrained_layout=True,
    )
    _, _, _, erp_ylim = plot_erp_panels(
        axes_b,
        epoch_data,
        args,
        title_prefix=channel_title("B. ERP", channel_set),
        erp_cluster_stats=erp_cluster_stats,
    )
    add_figure_stats_legend(fig_b, args, include_erp=True)
    save_figure(fig_b, output_dir / f"figure1_erp_{suffix}", args.dpi)
    if save_legacy_names:
        save_figure(fig_b, output_dir / "figure1_erp", args.dpi)

    fig_c, axes_c = plt.subplots(
        len(epoch_data.channel_names),
        1,
        figsize=(5.4, 6.5),
        constrained_layout=True,
    )
    pre_values, post_values, metric_ylim = plot_pre_post_panels(
        axes_c,
        epoch_data,
        args,
        title_prefix=channel_title("C. Pre vs Post", channel_set),
        pre_values=pre_values,
        post_values=post_values,
        paired_stats=pre_post_stats,
    )
    add_figure_stats_legend(fig_c, args, include_erp=False)
    save_figure(fig_c, output_dir / f"figure1_pre_post_{suffix}", args.dpi)
    if save_legacy_names:
        save_figure(fig_c, output_dir / "figure1_pre_post", args.dpi)

    plt.close(fig_b)
    plt.close(fig_c)

    return erp_ylim, metric_ylim, pre_values, post_values


def create_combined_figure(
    dataset: EEGLABContinuous,
    overview: OverviewData,
    epoch_data: EpochData,
    channel_set: ChannelSet,
    args: argparse.Namespace,
    output_dir: Path,
    save_legacy_names: bool,
    erp_cluster_stats: dict[str, list[ERPClusterStats]],
    pre_values: np.ndarray,
    post_values: np.ndarray,
    pre_post_stats: list[PairedChannelStats],
) -> None:
    """Create and save the combined publication/PPT-ready Figure 1."""

    fig = plt.figure(figsize=(13.5, 9.3), constrained_layout=True)
    outer = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.25],
        width_ratios=[1.35, 1.0],
    )

    ax_a = fig.add_subplot(outer[0, :])
    plot_overview_panel(ax_a, dataset, overview, args)

    gs_b = outer[1, 0].subgridspec(len(epoch_data.channel_names), 1)
    axes_b = [fig.add_subplot(gs_b[index, 0]) for index in range(len(epoch_data.channel_names))]
    plot_erp_panels(
        axes_b,
        epoch_data,
        args,
        title_prefix=channel_title("B. ERP", channel_set),
        erp_cluster_stats=erp_cluster_stats,
    )

    gs_c = outer[1, 1].subgridspec(len(epoch_data.channel_names), 1)
    axes_c = [fig.add_subplot(gs_c[index, 0]) for index in range(len(epoch_data.channel_names))]
    plot_pre_post_panels(
        axes_c,
        epoch_data,
        args,
        title_prefix=channel_title("C. Pre vs Post", channel_set),
        pre_values=pre_values,
        post_values=post_values,
        paired_stats=pre_post_stats,
    )

    suffix = channel_suffix(epoch_data.channel_names)
    add_figure_stats_legend(fig, args, include_erp=True)
    save_figure(fig, output_dir / f"figure1_combined_{suffix}", args.dpi)
    if save_legacy_names:
        save_figure(fig, output_dir / "figure1_combined", args.dpi)
    if args.show:
        plt.show()
    else:
        plt.close(fig)


def print_selection_summary(
    dataset: EEGLABContinuous,
    args: argparse.Namespace,
    overview: OverviewData,
    matched_events: list[EventRecord],
) -> None:
    """Print the selected condition and shared overview metadata."""

    subject = infer_subject(dataset, matched_events)
    print("\n=== Figure 1 selection ===")
    print(f"Subject: {subject}")
    print(f"Task: {args.task}")
    print(f"TMS: {args.tms}")
    print(f"Target: {args.target}")
    print(f"Category: {args.category}")
    print(f"Anchor: {args.anchor_role}")
    print("Overview channels:")
    print(", ".join(overview.channel_names))
    print(f"Figure 1A display filter: {overview.filter_text}")

    anchor = overview.anchor
    print("\n=== Figure 1A anchor event ===")
    print(pretty_event_id(anchor))
    print(f"Trial number: {display_value(anchor, 'trial_number') or 'n/a'}")
    print(f"Stimulus: {display_value(anchor, 'stimulus') or 'n/a'}")
    print(f"Phoneme1: {display_value(anchor, 'phoneme1') or 'n/a'}")
    print(f"Phoneme2: {display_value(anchor, 'phoneme2') or 'n/a'}")
    print(f"Phoneme3: {display_value(anchor, 'phoneme3') or 'n/a'}")
    print(f"Category: {display_value(anchor, 'category') or 'n/a'}")
    print(f"Onset: {anchor.onset_seconds:.6f} s")


def print_channel_set_summary(
    channel_set: ChannelSet,
    args: argparse.Namespace,
    epoch_data: EpochData,
) -> None:
    """Print the channel set, trial counts, and epoch windows."""

    pre_start, pre_end = comparison_pre_window(args)
    print("\n=== Figure 1 channel set ===")
    print(f"Mode: {channel_set.mode}")
    print(f"Channels: {', '.join(epoch_data.channel_names)}")
    if channel_set.title_note:
        print(f"Purpose: {channel_set.title_note}")
    print(f"Matched anchors before de-duplication: {epoch_data.raw_match_count}")
    print(f"Duplicate anchors skipped: {epoch_data.duplicate_count}")
    print(f"Boundary-excluded anchors: {epoch_data.boundary_excluded_count}")
    print(f"Matched trials: {len(epoch_data.events)}")
    print(f"Event ID field used: {epoch_data.event_id_field}")
    print(f"Epoch: {-args.pre:g} ~ {args.post:g} s")
    print(f"Baseline: {args.baseline_start:g} ~ {args.baseline_end:g} s")
    metric_name = "RMS" if args.comparison_metric == "rms" else "Mean"
    print(f"Pre {metric_name}: {pre_start:g} ~ {pre_end:g} s")
    print(f"Post {metric_name}: {args.post_start:g} ~ {args.post_end:g} s")
    print(
        "ERP stats: "
        f"{args.erp_stats_start:g} ~ {args.erp_stats_end:g} s, "
        f"{args.erp_permutations:,} sign-flip permutations"
    )
    print(f"Pre/Post stats: {args.stats_test}, Holm-corrected across channels")


def print_quality_summary(
    epoch_data: EpochData,
    erp_ylim: tuple[float, float],
    metric_ylim: tuple[float, float],
    pre_values: np.ndarray,
    post_values: np.ndarray,
    args: argparse.Namespace,
    channel_set: ChannelSet,
) -> None:
    """Print compact plotting scale and amplitude diagnostics."""

    mean, lower, upper = erp_mean_ci(epoch_data.epochs_uv)
    print(f"\n=== Figure 1 diagnostics ({channel_set.mode}) ===")
    print(f"ERP y-limit shared across channels: {erp_ylim[0]:.3f} to {erp_ylim[1]:.3f} uV")
    print(
        "ERP mean range across channels/time: "
        f"{float(mean.min()):.3f} to {float(mean.max()):.3f} uV"
    )
    print(
        "ERP 95% CI envelope range: "
        f"{float(lower.min()):.3f} to {float(upper.max()):.3f} uV"
    )
    metric_name = "RMS" if args.comparison_metric == "rms" else "Mean"
    print(
        f"{metric_name} y-limit shared across channels: "
        f"{metric_ylim[0]:.3f} to {metric_ylim[1]:.3f} uV"
    )
    for channel_index, channel_name in enumerate(epoch_data.channel_names):
        pre_mean = float(pre_values[:, channel_index].mean())
        post_mean = float(post_values[:, channel_index].mean())
        delta = post_mean - pre_mean
        print(
            f"{channel_name}: pre {pre_mean:.3f} uV, "
            f"post {post_mean:.3f} uV, post-pre {delta:.3f} uV"
        )


def main() -> None:
    """Run Figure 1 generation."""

    args = parse_args()
    validate_args(args)
    channel_sets = resolve_channel_sets(args)
    set_publication_style()

    dataset = load_eeglab_metadata(args.set)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Dataset ===")
    print(f"SET: {dataset.set_path}")
    print(f"FDT: {dataset.fdt_path or 'embedded in .set'}")
    print(f"Channels: {dataset.n_channels}")
    print(f"Samples: {dataset.n_samples:,}")
    print(f"Sampling rate: {dataset.sfreq:g} Hz")
    print(f"Duration: {dataset.duration_seconds:.3f} s")
    print(f"Events: {len(dataset.events):,}")

    raw_matches = select_anchor_events(dataset, args)
    overview = prepare_overview_data(dataset, args, raw_matches)
    print_selection_summary(dataset, args, overview, raw_matches)
    create_overview_figure(dataset, overview, args, output_dir)

    for channel_set in channel_sets:
        epoch_data = prepare_epoch_data(
            dataset,
            args,
            channel_set.channels,
            raw_matches=raw_matches,
        )
        print_channel_set_summary(channel_set, args, epoch_data)

        suffix = channel_suffix(epoch_data.channel_names)
        erp_cluster_stats = compute_erp_cluster_statistics(epoch_data, args)
        pre_values, post_values = compute_pre_post_values(epoch_data, args)
        pre_post_stats = compute_paired_statistics(
            pre_values,
            post_values,
            epoch_data.channel_names,
            args,
        )
        save_erp_cluster_stats_csv(
            output_dir / f"figure1_erp_cluster_stats_{suffix}.csv",
            epoch_data,
            erp_cluster_stats,
        )
        save_pre_post_stats_csv(
            output_dir / f"figure1_pre_post_stats_{suffix}.csv",
            pre_post_stats,
        )
        print_erp_cluster_statistics(
            erp_cluster_stats,
            epoch_data.channel_names,
            args,
        )
        print_pre_post_statistics(pre_post_stats)

        save_legacy_names = channel_set.mode == "motor"
        erp_ylim, metric_ylim, pre_values, post_values = create_channel_figures(
            epoch_data,
            channel_set,
            args,
            output_dir,
            save_legacy_names=save_legacy_names,
            erp_cluster_stats=erp_cluster_stats,
            pre_values=pre_values,
            post_values=post_values,
            pre_post_stats=pre_post_stats,
        )
        print_quality_summary(
            epoch_data,
            erp_ylim,
            metric_ylim,
            pre_values,
            post_values,
            args,
            channel_set,
        )

        create_combined_figure(
            dataset,
            overview,
            epoch_data,
            channel_set,
            args,
            output_dir,
            save_legacy_names=save_legacy_names,
            erp_cluster_stats=erp_cluster_stats,
            pre_values=pre_values,
            post_values=post_values,
            pre_post_stats=pre_post_stats,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
