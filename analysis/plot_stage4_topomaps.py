#!/usr/bin/env python3

"""
plot_stage4_topomaps.py

Create scalp EEG topographic maps directly from continuous Stage-4 EEGLAB data.

Default maps:
    1. Baseline       : -200 ~    0 ms relative to stim_1
    2. Trial onset    :    0 ~   50 ms relative to stim_1
    3. Post onset     :   75 ~  125 ms relative to stim_1
    4. Trial final    :  900 ~ 1000 ms relative to stim_1

The script:
    - loads a continuous EEGLAB .set file
    - finds stim_1 events
    - optionally filters task / TMS / target / category
    - extracts short windows without permanently epoching the dataset
    - averages within each trial
    - averages across trials
    - plots four topographic maps with one common color scale
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import mne


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def normalize_name(x: str) -> str:
    """Normalize text for matching."""
    return (
        str(x)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def is_non_scalp_channel(name: str) -> bool:
    """
    Return True for channels that should not be used in scalp topography.

    Adjust this if your channel naming convention changes.
    """
    n = name.upper().strip()

    if n in {"M1", "M2"}:
        return True

    if "EOG" in n:
        return True

    if n.startswith("BIP"):
        return True

    return False


def remove_non_scalp_channels(raw: mne.io.BaseRaw) -> None:
    """Drop EOG, mastoid, and BIP channels by name."""
    drop = [
        ch
        for ch in raw.ch_names
        if is_non_scalp_channel(ch)
    ]

    if drop:
        print("\nDropping non-scalp channels:")
        print("   " + ", ".join(drop))
        raw.drop_channels(drop)


def get_valid_location_count(raw: mne.io.BaseRaw) -> tuple[int, int]:
    """Count EEG channels with usable XYZ positions."""

    eeg_picks = mne.pick_types(
        raw.info,
        eeg=True,
        exclude=[]
    )

    valid = 0

    for idx in eeg_picks:
        xyz = np.asarray(
            raw.info["chs"][idx]["loc"][:3],
            dtype=float
        )

        if (
            np.all(np.isfinite(xyz))
            and np.linalg.norm(xyz) > 1e-8
        ):
            valid += 1

    return valid, len(eeg_picks)


def import_locations_from_reference(
    raw: mne.io.BaseRaw,
    reference_path: str
) -> None:
    """
    Import electrode locations from another EEGLAB .set file.

    Channels are matched by channel name.
    """

    print("\nLoading channel-location reference:")
    print(f"   {reference_path}")

    ref = mne.io.read_raw_eeglab(
        reference_path,
        preload=False,
        verbose="warning"
    )

    remove_non_scalp_channels(ref)

    montage = ref.get_montage()

    if montage is None:
        raise RuntimeError(
            "Reference dataset does not contain an MNE-readable montage."
        )

    raw.set_montage(
        montage,
        match_case=False,
        on_missing="ignore",
        verbose=False
    )

    valid, total = get_valid_location_count(raw)

    print(
        f"Imported usable EEG locations: "
        f"{valid}/{total}"
    )


def check_locations(
    raw: mne.io.BaseRaw,
    loc_reference: str | None
) -> None:
    """Ensure there are sufficient positions for topomap plotting."""

    valid, total = get_valid_location_count(raw)

    print(
        f"\nEEG channels with usable positions: "
        f"{valid}/{total}"
    )

    # Require most channels to have positions.
    enough = total > 0 and valid >= max(3, int(total * 0.8))

    if enough:
        return

    if loc_reference is None:
        raise RuntimeError(
            "\nInsufficient electrode locations.\n"
            "Use:\n"
            "    --loc-reference path/to/reference.set\n"
            "with a dataset that contains valid channel locations."
        )

    import_locations_from_reference(
        raw,
        loc_reference
    )

    valid, total = get_valid_location_count(raw)

    if valid < max(3, int(total * 0.8)):
        raise RuntimeError(
            f"Still insufficient electrode locations "
            f"after reference import: {valid}/{total}"
        )


# ---------------------------------------------------------------------
# Event handling
# ---------------------------------------------------------------------

def parse_event_description(description: str):
    """
    Parse event label.

    Expected format:

        task/tms/target/category/event_role

    Example:

        words/tms0/control/real/stim_1
        words/tms1/ba44/nonce/stim_1
    """

    parts = [
        normalize_name(x)
        for x in str(description).split("/")
    ]

    if len(parts) < 5:
        return None

    return {
        "task": parts[0],
        "tms": parts[1],
        "target": parts[2],
        "category": parts[3],
        "role": parts[-1],
    }


def event_matches(
    description: str,
    task: str,
    tms: str,
    target: str,
    category: str
) -> bool:
    """Check whether an event matches requested filters."""

    info = parse_event_description(description)

    if info is None:
        return False

    if info["role"] != "stim_1":
        return False

    requested = {
        "task": normalize_name(task),
        "tms": normalize_name(tms),
        "target": normalize_name(target),
        "category": normalize_name(category),
    }

    for key in ("task", "tms", "target", "category"):

        if requested[key] in {"all", "*"}:
            continue

        if info[key] != requested[key]:
            return False

    return True


def find_trial_onsets(
    raw: mne.io.BaseRaw,
    task: str,
    tms: str,
    target: str,
    category: str
) -> list[float]:

    selected = []

    for onset, description in zip(
        raw.annotations.onset,
        raw.annotations.description
    ):

        if event_matches(
            description,
            task=task,
            tms=tms,
            target=target,
            category=category
        ):
            selected.append(float(onset))

    return selected


def print_available_stim1_events(raw: mne.io.BaseRaw) -> None:

    descriptions = sorted(
        {
            str(d)
            for d in raw.annotations.description
            if normalize_name(str(d)).endswith("stim_1")
        }
    )

    print("\nAvailable stim_1 event labels:")

    for d in descriptions:
        print(f"   {d}")


# ---------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------

def extract_window(
    raw: mne.io.BaseRaw,
    onset_seconds: float,
    window_ms: tuple[float, float],
    picks
):
    """
    Extract one trial/window.

    Returns channel mean over the requested time window.
    """

    start_ms, end_ms = window_ms

    start_time = onset_seconds + start_ms / 1000.0
    end_time = onset_seconds + end_ms / 1000.0

    # Skip trials whose requested window extends outside recording.
    if start_time < 0:
        return None

    if end_time > raw.times[-1]:
        return None

    start_sample, end_sample = raw.time_as_index(
        [start_time, end_time],
        use_rounding=True
    )

    if end_sample <= start_sample:
        end_sample = start_sample + 1

    data = raw.get_data(
        picks=picks,
        start=start_sample,
        stop=end_sample
    )

    if data.size == 0:
        return None

    # mean over time → one amplitude value per channel
    return np.nanmean(data, axis=1)


def calculate_topographies(
    raw,
    trial_onsets,
    windows,
    baseline_correct=False
):
    """Calculate averaged topographies across trials."""

    picks = mne.pick_types(
        raw.info,
        eeg=True,
        exclude="bads"
    )

    results = {
        name: []
        for name in windows
    }

    skipped = 0

    for onset in trial_onsets:

        trial_vectors = {}

        valid_trial = True

        for name, window in windows.items():

            vector = extract_window(
                raw,
                onset,
                window,
                picks
            )

            if vector is None:
                valid_trial = False
                break

            trial_vectors[name] = vector

        if not valid_trial:
            skipped += 1
            continue

        # Optional ERP-style baseline correction.
        #
        # Baseline map itself remains the actual baseline amplitude,
        # while post-stimulus maps are converted to change from baseline.
        if baseline_correct:

            baseline = trial_vectors["Baseline"].copy()

            for name in trial_vectors:

                if name != "Baseline":
                    trial_vectors[name] = (
                        trial_vectors[name] - baseline
                    )

        for name, vector in trial_vectors.items():
            results[name].append(vector)

    print(f"\nTrials excluded because of recording boundaries: {skipped}")

    averaged = {}

    for name, values in results.items():

        if not values:
            raise RuntimeError(
                f"No valid data were extracted for: {name}"
            )

        values = np.stack(values, axis=0)

        print(
            f"{name:16s}: "
            f"{values.shape[0]} trials"
        )

        averaged[name] = np.nanmean(
            values,
            axis=0
        )

    # Keep info containing only plotted EEG channels.
    info = mne.pick_info(
        raw.info,
        picks,
        copy=True
    )

    return averaged, info


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_topographies(
    averaged,
    info,
    output_path,
    baseline_correct=False,
    show=False
):

    # MNE stores EEG in volts.
    # Convert to microvolts for plotting.
    maps_uv = {
        name: data * 1e6
        for name, data in averaged.items()
    }

    '''
    all_values = np.concatenate(
        list(maps_uv.values())
    )

    vmax = np.nanmax(
        np.abs(all_values)
    )

    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    vlim = (-vmax, vmax)
    '''
    
    vlim = (-40, 40)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(15, 4.5)
    )

    titles = {
        "Baseline":
            "Pre-stimulus\n−200–0 ms",

        "Trial onset":
            "Stimulus onset\n0–50 ms",

        "Post 100 ms":
            "~100 ms post-stimulus\n75–125 ms",

        "Final":
            "Last post-stimulus\n900–1000 ms",
    }

    image = None

    for ax, name in zip(axes, averaged.keys()):

        image, _ = mne.viz.plot_topomap(
            maps_uv[name],
            info,
            axes=ax,
            show=False,
            sensors=True,
            contours=6,
            cmap="RdBu_r",
            vlim=vlim,
            outlines="head",
            sphere="auto"
        )
        
        # ---------------------------------------------------------
        # Enlarge ONLY the black circular head outline
        # ---------------------------------------------------------

        # ax.lines 안에서 점 개수가 가장 많은 선이 head circle
        # ---------------------------------------------------------
        # Adjust head outline ONLY
        # - enlarge head to 1.25
        # - move head upward
        # - move/scale nose + ears together
        # - sensors/topomap/contours remain unchanged
        # ---------------------------------------------------------

        HEAD_SCALE = 1.25
        HEAD_Y_SHIFT = 0.012   # positive = upward

        # solid lines = head circle + nose + ears
        # sensor dots have linestyle="None", so they are excluded
        outline_lines = [
            line for line in ax.lines
            if line.get_linestyle() not in ("None", "", " ")
        ]

        # 가장 많은 좌표를 가진 solid line = head circle
        head_line = max(
            outline_lines,
            key=lambda line: len(line.get_xdata())
        )

        # ORIGINAL head center
        head_x = np.asarray(head_line.get_xdata(), dtype=float)
        head_y = np.asarray(head_line.get_ydata(), dtype=float)

        cx = np.mean(head_x)
        cy = np.mean(head_y)

        # head + nose + left ear + right ear
        # 모두 같은 중심 기준으로 확대하고 위로 이동
        for line in outline_lines:

            x = np.asarray(line.get_xdata(), dtype=float)
            y = np.asarray(line.get_ydata(), dtype=float)

            line.set_xdata(
                cx + (x - cx) * HEAD_SCALE
            )

            line.set_ydata(
                cy + (y - cy) * HEAD_SCALE
                + HEAD_Y_SHIFT
            )

        ax.set_title(
            titles[name],
            fontsize=12,
            y=1.12
        )

    colorbar = fig.colorbar(
        image,
        ax=axes,
        shrink=0.82,
        pad=0.03
    )

    if baseline_correct:
        colorbar.set_label(
            "Amplitude / baseline change (µV)"
        )
    else:
        colorbar.set_label(
            "Amplitude (µV)"
        )

    fig.suptitle(
        "EEG scalp topography",
        fontsize=15
    )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print("\nSaved figure:")
    print(f"   {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Plot Stage-4 continuous EEG topographies "
            "relative to stim_1."
        )
    )

    parser.add_argument(
        "--set",
        required=True,
        help="Stage-4 continuous EEGLAB .set file"
    )

    parser.add_argument(
        "--task",
        default="words",
        help="Task filter. Default: words"
    )

    parser.add_argument(
        "--tms",
        default="all",
        help="tms0, tms1, or all"
    )

    parser.add_argument(
        "--target",
        default="all",
        help="control, lip, tongue, ba06, ba44, or all"
    )

    parser.add_argument(
        "--category",
        default="all",
        help="real, nonce, etc. Default: all"
    )

    parser.add_argument(
        "--loc-reference",
        default=None,
        help=(
            "Optional .set file containing valid "
            "electrode locations"
        )
    )

    parser.add_argument(
        "--average-reference",
        action="store_true",
        help="Apply average EEG reference before plotting"
    )

    parser.add_argument(
        "--baseline-correct",
        action="store_true",
        help=(
            "Subtract each trial's baseline scalp values "
            "from onset/post/final maps"
        )
    )

    parser.add_argument(
        "--output",
        default="./figures/stage4_topomaps.png",
        help="Output PNG path"
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure"
    )

    args = parser.parse_args()

    print("\n==========================================")
    print(" STAGE-4 EEG TOPOGRAPHY")
    print("==========================================")

    print("\nLoading:")
    print(f"   {args.set}")

    raw = mne.io.read_raw_eeglab(
        args.set,
        preload=True,
        verbose="warning"
    )

    print(
        f"\nSampling frequency: "
        f"{raw.info['sfreq']:.3f} Hz"
    )

    print(
        f"Recording duration: "
        f"{raw.times[-1]:.2f} sec"
    )

    print(
        f"Original channels: "
        f"{len(raw.ch_names)}"
    )

    print(
        f"Annotations/events: "
        f"{len(raw.annotations)}"
    )

    # --------------------------------------------------------------
    # Remove channels that should not enter scalp topography
    # --------------------------------------------------------------

    remove_non_scalp_channels(raw)

    # --------------------------------------------------------------
    # Check/import electrode locations
    # --------------------------------------------------------------

    check_locations(
        raw,
        args.loc_reference
    )

    # --------------------------------------------------------------
    # Optional re-reference
    # --------------------------------------------------------------

    if args.average_reference:

        print("\nApplying average EEG reference.")

        raw.set_eeg_reference(
            "average",
            projection=False,
            verbose=False
        )

    # --------------------------------------------------------------
    # Find selected trials
    # --------------------------------------------------------------

    trial_onsets = find_trial_onsets(
        raw,
        task=args.task,
        tms=args.tms,
        target=args.target,
        category=args.category
    )

    print("\nSelection:")
    print(f"   task     = {args.task}")
    print(f"   tms      = {args.tms}")
    print(f"   target   = {args.target}")
    print(f"   category = {args.category}")

    print(
        f"\nSelected stim_1 trials: "
        f"{len(trial_onsets)}"
    )

    if len(trial_onsets) == 0:

        print_available_stim1_events(raw)

        raise RuntimeError(
            "\nNo matching stim_1 trials were found."
        )

    # --------------------------------------------------------------
    # Time windows relative to stim_1
    # --------------------------------------------------------------

    windows = {
        "Baseline": (-200, 0),
        "Trial onset": (0, 50),
        "Post 100 ms": (75, 125),
        "Final": (900, 1000),
    }

    print("\nTime windows relative to stim_1:")

    for name, (start, end) in windows.items():
        print(
            f"   {name:16s}: "
            f"{start:6.0f} ~ {end:6.0f} ms"
        )

    # --------------------------------------------------------------
    # Extract and average
    # --------------------------------------------------------------

    averaged, info = calculate_topographies(
        raw,
        trial_onsets,
        windows,
        baseline_correct=args.baseline_correct
    )

    # --------------------------------------------------------------
    # Plot
    # --------------------------------------------------------------

    plot_topographies(
        averaged,
        info,
        output_path=args.output,
        baseline_correct=args.baseline_correct,
        show=args.show
    )


if __name__ == "__main__":
    main()