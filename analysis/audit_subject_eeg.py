#!/usr/bin/env python3
"""
audit_subject_eeg.py

S01-S16 음소 분류 데이터에서 다음을 한 번에 점검합니다.

1) 피험자별/음소별 trial 수
2) 빈 라벨, 중복 trial ID, class 누락/불균형
3) confusion matrix (prediction 파일이 있을 때)
4) EEG 품질 지표
   - NaN/Inf
   - flat channel
   - peak-to-peak amplitude
   - RMS
   - 급격한 jump/electrode-pop
   - low-frequency drift
   - high-frequency muscle noise
   - 59-61 Hz line noise
   - common-mode/global artifact
   - kurtosis
5) 이상 trial/channel 목록
6) S06/S08/S10과 전체 피험자 비교 요약

권장 입력 NPZ 구조
------------------
필수:
    X: (trials, channels, timepoints)

라벨 키 예:
    y_phoneme, phoneme1_per_trial, phoneme1, labels, y

선택:
    sfreq / sampling_frequency
    ch_names / channel_names
    TrialN / trial_number / trial_id
    subject_per_trial

Prediction 입력은 다음 중 하나:
A. CSV:
    subject,y_true,y_pred
    S08,ba,pa
    ...

B. combined NPZ:
    subject_per_trial, y_true, y_pred
    선택적으로 y_prob 또는 probabilities

C. 피험자별 prediction NPZ 폴더:
    S08_predictions.npz 안에 y_true, y_pred
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis
warnings.filterwarnings(
    "ignore",
    message="Mean of empty slice",
    category=RuntimeWarning,
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

SUBJECT_RE = re.compile(r"(?<![A-Za-z0-9])(S\d{2})(?![A-Za-z0-9])", re.IGNORECASE)

X_CANDIDATES = ("X", "x", "eeg", "data", "epochs")
LABEL_CANDIDATES = (
    "y_phoneme",
    "phoneme1_per_trial",
    "phoneme_per_trial",
    "phoneme1",
    "Phoneme1",
    "phoneme",
    "labels",
    "label",
    "target",
    "targets",
    "y",
)
SFREQ_CANDIDATES = (
    "sfreq",
    "sampling_frequency",
    "sample_frequency",
    "srate",
    "fs",
)
CHANNEL_NAME_CANDIDATES = (
    "ch_names",
    "channel_names",
    "channels",
    "chan_names",
)
TRIAL_ID_CANDIDATES = (
    "TrialN",
    "trial_number",
    "trial_numbers",
    "trial_id",
    "trial_ids",
    "epoch_in_source_file",
)
SUBJECT_KEY_CANDIDATES = (
    "subject_per_trial",
    "subjects",
    "subject",
)
PRED_TRUE_CANDIDATES = ("y_true", "true", "labels", "target", "targets", "y")
PRED_PRED_CANDIDATES = ("y_pred", "pred", "predictions", "prediction")
PRED_PROB_CANDIDATES = (
    "y_prob",
    "y_proba",
    "probabilities",
    "probs",
    "prediction_probabilities",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Audit EEG phoneme data quality, labels, trial counts, and confusion matrices.",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("."),
        help="S01-S16 NPZ 파일이 들어 있는 폴더. 하위 폴더도 재귀 탐색합니다.",
    )
    p.add_argument(
        "--combined-data-npz",
        type=Path,
        default=None,
        help=(
            "모든 피험자가 하나의 NPZ에 있고 subject_per_trial로 구분되는 경우 사용. "
            "이 옵션을 주면 --data-dir의 피험자별 파일 탐색 대신 이 파일을 사용합니다."
        ),
    )
    p.add_argument(
        "--subjects",
        nargs="+",
        default=[f"S{i:02d}" for i in range(1, 17)],
        help="분석할 피험자.",
    )
    p.add_argument(
        "--focus-subjects",
        nargs="+",
        default=["S06", "S08", "S10"],
        help="최종 비교에서 강조할 피험자.",
    )
    p.add_argument(
        "--file-pattern",
        default="*.npz",
        help="데이터 파일 검색 패턴.",
    )
    p.add_argument(
        "--x-key",
        default=None,
        help="NPZ의 EEG 배열 키. 생략하면 자동 탐색.",
    )
    p.add_argument(
        "--label-key",
        default=None,
        help="NPZ의 음소 라벨 키. 생략하면 자동 탐색.",
    )
    p.add_argument(
        "--trial-id-key",
        default=None,
        help="trial ID 키. 생략하면 자동 탐색.",
    )
    p.add_argument(
        "--sfreq",
        type=float,
        default=None,
        help="샘플링 주파수. NPZ에 없을 때 사용.",
    )
    p.add_argument(
        "--pred-csv",
        type=Path,
        default=None,
        help="subject,y_true,y_pred 열을 가진 combined prediction CSV.",
    )
    p.add_argument(
        "--pred-npz",
        type=Path,
        default=None,
        help="subject_per_trial,y_true,y_pred를 가진 combined prediction NPZ.",
    )
    p.add_argument(
        "--pred-dir",
        type=Path,
        default=None,
        help="피험자별 prediction NPZ 폴더.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eeg_subject_audit"),
        help="결과 저장 폴더.",
    )
    p.add_argument(
        "--max-welch-trials",
        type=int,
        default=400,
        help="피험자당 spectral metric을 계산할 최대 trial 수. 초과 시 균등 샘플링.",
    )
    p.add_argument(
        "--bad-z",
        type=float,
        default=4.0,
        help="피험자 내부 robust z-score 이상 trial 기준.",
    )
    p.add_argument(
        "--subject-outlier-z",
        type=float,
        default=3.5,
        help="피험자 간 outlier 기준.",
    )
    p.add_argument(
        "--unit",
        choices=("auto", "V", "uV"),
        default="auto",
        help="EEG amplitude 단위.",
    )
    return p.parse_args()


def decode_strings(arr: Any) -> np.ndarray:
    """bytes/object/string/integer 배열을 비교 가능한 1-D 문자열 배열로 변환."""
    x = np.asarray(arr)
    if x.ndim == 0:
        x = x.reshape(1)
    x = x.reshape(-1)

    out: list[str] = []
    for value in x:
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        elif isinstance(value, np.bytes_):
            value = value.tobytes().decode("utf-8", errors="replace")

        if isinstance(value, (float, np.floating)) and np.isnan(value):
            out.append("")
        else:
            s = str(value).strip()
            if s.lower() in {"nan", "none", "<na>"}:
                s = ""
            out.append(s)
    return np.asarray(out, dtype=object)


def first_existing_key(
    npz: np.lib.npyio.NpzFile,
    explicit: str | None,
    candidates: Iterable[str],
    required: bool = True,
) -> str | None:
    if explicit:
        if explicit not in npz.files:
            raise KeyError(
                f"Requested key '{explicit}' is missing. Available keys: {npz.files}"
            )
        return explicit

    lower_map = {k.lower(): k for k in npz.files}
    for candidate in candidates:
        if candidate in npz.files:
            return candidate
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    if required:
        raise KeyError(
            f"Could not find any of {tuple(candidates)}. Available keys: {npz.files}"
        )
    return None


def infer_subject(path: Path) -> str | None:
    for text in (path.stem, path.parent.name, str(path)):
        m = SUBJECT_RE.search(text)
        if m:
            return m.group(1).upper()
    return None


def find_subject_files(
    data_dir: Path,
    subjects: list[str],
    pattern: str,
) -> dict[str, Path]:
    files = sorted(data_dir.rglob(pattern))
    found: dict[str, list[Path]] = {s.upper(): [] for s in subjects}

    for path in files:
        subject = infer_subject(path)
        if subject in found:
            found[subject].append(path)

    selected: dict[str, Path] = {}
    for subject, candidates in found.items():
        if not candidates:
            continue

        # prediction/report 파일보다는 실제 EEG 데이터처럼 보이는 파일을 우선.
        def rank(p: Path) -> tuple[int, int, str]:
            name = p.name.lower()
            penalty = 0
            if any(token in name for token in ("pred", "result", "metric", "report")):
                penalty += 100
            if "_7_" in name:
                penalty -= 5
            return (penalty, len(str(p)), str(p))

        candidates = sorted(candidates, key=rank)
        selected[subject] = candidates[0]

        if len(candidates) > 1:
            print(
                f"[{subject}] Multiple files found; using:\n"
                f"    {selected[subject]}\n"
                f"    Other candidates: {len(candidates) - 1}"
            )

    return selected


def orient_x_to_trials(
    X: np.ndarray,
    n_labels: int,
) -> np.ndarray:
    """
    X를 (trials, channels, timepoints)로 맞춤.
    3차원만 허용합니다.
    """
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"X must be 3-D, got shape={X.shape}")

    matching_axes = [axis for axis, size in enumerate(X.shape) if size == n_labels]
    if not matching_axes:
        raise ValueError(
            f"No X axis matches n_labels={n_labels}. X shape={X.shape}"
        )

    trial_axis = matching_axes[0]
    if trial_axis != 0:
        X = np.moveaxis(X, trial_axis, 0)

    # 남은 두 축 중 보통 channel 수가 더 작습니다.
    # (trials, time, channels)인 경우 자동 교환.
    if X.shape[1] > X.shape[2] and X.shape[2] <= 512:
        X = np.transpose(X, (0, 2, 1))

    return X


def get_scalar(npz: np.lib.npyio.NpzFile, key: str | None) -> float | None:
    if key is None:
        return None
    value = np.asarray(npz[key]).reshape(-1)
    if value.size == 0:
        return None
    try:
        return float(value[0])
    except (TypeError, ValueError):
        return None


def to_microvolts(X: np.ndarray, unit: str) -> tuple[np.ndarray, str]:
    """
    auto:
      EEG가 volts이면 보통 중앙 절대값이 0.01보다 훨씬 작음.
      이미 microvolts이면 보통 수~수백 범위.
    """
    finite = np.abs(X[np.isfinite(X)])
    median_abs = float(np.median(finite)) if finite.size else np.nan

    if unit == "V":
        return X * 1e6, "V -> uV"
    if unit == "uV":
        return X, "uV"

    if np.isfinite(median_abs) and median_abs < 0.01:
        return X * 1e6, f"auto: V -> uV (median |x|={median_abs:.3g})"
    return X, f"auto: treated as uV (median |x|={median_abs:.3g})"


def robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    z = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if valid.sum() < 3:
        return z

    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    if mad <= 1e-12:
        std = np.nanstd(values)
        if std <= 1e-12:
            z[valid] = 0.0
        else:
            z[valid] = (values[valid] - med) / std
    else:
        z[valid] = 0.67448975 * (values[valid] - med) / mad
    return z


def band_power_from_psd(
    freqs: np.ndarray,
    psd: np.ndarray,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    mask = (freqs >= fmin) & (freqs <= fmax)
    n_bins = int(mask.sum())
    if n_bins < 1:
        return np.full(psd.shape[:-1], np.nan)
    if n_bins == 1:
        if len(freqs) >= 2:
            df = float(np.nanmedian(np.diff(freqs)))
        else:
            df = 1.0
        return psd[..., mask][..., 0] * df
    return np.trapezoid(psd[..., mask], freqs[mask], axis=-1)


def compute_trial_metrics(
    X_uv: np.ndarray,
    sfreq: float,
    max_welch_trials: int,
) -> pd.DataFrame:
    n_trials, n_channels, n_times = X_uv.shape
    eps = 1e-12

    finite_X = np.where(np.isfinite(X_uv), X_uv, np.nan)

    nan_fraction = np.mean(~np.isfinite(X_uv), axis=(1, 2))
    channel_std = np.nanstd(finite_X, axis=2)
    subject_typical_std = np.nanmedian(channel_std)
    flat_threshold = max(subject_typical_std * 1e-4, 1e-6)
    flat_channel_fraction = np.mean(channel_std < flat_threshold, axis=1)

    ptp_by_channel = np.nanmax(finite_X, axis=2) - np.nanmin(finite_X, axis=2)
    p2p_uv = np.nanmedian(ptp_by_channel, axis=1)
    max_p2p_uv = np.nanmax(ptp_by_channel, axis=1)

    rms_by_channel = np.sqrt(np.nanmean(finite_X**2, axis=2))
    rms_uv = np.nanmedian(rms_by_channel, axis=1)
    max_abs_uv = np.nanmax(np.abs(finite_X), axis=(1, 2))

    dx = np.diff(finite_X, axis=2)
    robust_dx_scale = np.nanmedian(np.abs(dx), axis=(1, 2)) + eps
    jump_ratio = np.nanmax(np.abs(dx), axis=(1, 2)) / robust_dx_scale

    # across-channel mean이 전체 신호 분산을 얼마나 설명하는지:
    # 전극 전체가 동시에 크게 움직이는 global/movement artifact에 민감.
    common_signal = np.nanmean(finite_X, axis=1)
    common_var = np.nanvar(common_signal, axis=1)
    median_channel_var = np.nanmedian(np.nanvar(finite_X, axis=2), axis=1)
    common_mode_ratio = common_var / (median_channel_var + eps)

    median_kurtosis = np.nanmedian(
        kurtosis(finite_X, axis=2, fisher=True, nan_policy="omit"),
        axis=1,
    )

    metrics = pd.DataFrame(
        {
            "trial_index": np.arange(n_trials),
            "nan_fraction": nan_fraction,
            "flat_channel_fraction": flat_channel_fraction,
            "p2p_uv": p2p_uv,
            "max_p2p_uv": max_p2p_uv,
            "rms_uv": rms_uv,
            "max_abs_uv": max_abs_uv,
            "jump_ratio": jump_ratio,
            "common_mode_ratio": common_mode_ratio,
            "median_kurtosis": median_kurtosis,
            "low_freq_ratio": np.nan,
            "high_freq_ratio": np.nan,
            "line_noise_ratio": np.nan,
        }
    )

    # Welch는 계산량이 크므로 균등 샘플링.
    if n_trials <= max_welch_trials:
        spectral_indices = np.arange(n_trials)
    else:
        spectral_indices = np.unique(
            np.linspace(0, n_trials - 1, max_welch_trials).astype(int)
        )

    # NaN을 채널별 중앙값으로 대체.
    X_spec = finite_X[spectral_indices].copy()
    channel_medians = np.nanmedian(X_spec, axis=2, keepdims=True)
    channel_medians = np.nan_to_num(channel_medians, nan=0.0)
    X_spec = np.where(np.isfinite(X_spec), X_spec, channel_medians)

    nperseg = min(n_times, max(64, int(round(sfreq * 0.5))))
    freqs, psd = welch(
        X_spec,
        fs=sfreq,
        axis=2,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
    )
    # 채널 중앙값 PSD -> trial별 spectrum
    trial_psd = np.nanmedian(psd, axis=1)

    base_power = band_power_from_psd(freqs, trial_psd, 1.0, min(30.0, sfreq / 2 - 0.5))
    low_power = band_power_from_psd(freqs, trial_psd, 0.1, min(1.0, sfreq / 2 - 0.5))
    high_power = band_power_from_psd(freqs, trial_psd, 30.0, min(50.0, sfreq / 2 - 0.5))

    low_ratio = low_power / (base_power + eps)
    high_ratio = high_power / (base_power + eps)

    if sfreq / 2 >= 61.0:
        line_power = band_power_from_psd(freqs, trial_psd, 59.0, 61.0)
        broad_power = band_power_from_psd(freqs, trial_psd, 1.0, min(100.0, sfreq / 2 - 0.5))
        line_ratio = line_power / (broad_power + eps)
    else:
        line_ratio = np.full(len(spectral_indices), np.nan)

    metrics.loc[spectral_indices, "low_freq_ratio"] = low_ratio
    metrics.loc[spectral_indices, "high_freq_ratio"] = high_ratio
    metrics.loc[spectral_indices, "line_noise_ratio"] = line_ratio

    # sampling되지 않은 trial은 subject median으로 채움.
    for col in ("low_freq_ratio", "high_freq_ratio", "line_noise_ratio"):
        med = metrics[col].median(skipna=True)
        if np.isfinite(med):
            metrics[col] = metrics[col].fillna(med)

    return metrics


def add_bad_trial_scores(metrics: pd.DataFrame, bad_z: float) -> pd.DataFrame:
    out = metrics.copy()

    z_columns = {
        "p2p_uv": "z_p2p",
        "max_p2p_uv": "z_max_p2p",
        "rms_uv": "z_rms",
        "max_abs_uv": "z_max_abs",
        "jump_ratio": "z_jump",
        "common_mode_ratio": "z_common_mode",
        "median_kurtosis": "z_kurtosis",
        "low_freq_ratio": "z_low_freq",
        "high_freq_ratio": "z_high_freq",
        "line_noise_ratio": "z_line_noise",
    }

    for source, target in z_columns.items():
        values = out[source].to_numpy(float)
        # 양수 방향 이상치만 중요하므로 로그 변환 가능한 지표는 로그 후 robust z.
        if np.all(values[np.isfinite(values)] >= 0):
            z = robust_z(np.log1p(values))
        else:
            z = robust_z(values)
        out[target] = z

    z_names = list(z_columns.values())
    positive_excess = np.column_stack(
        [
            np.maximum(np.nan_to_num(out[c].to_numpy(float), nan=0.0) - bad_z, 0.0)
            for c in z_names
        ]
    )
    out["bad_score"] = positive_excess.sum(axis=1)

    hard_bad = (
        (out["nan_fraction"] > 0)
        | (out["flat_channel_fraction"] >= 0.10)
        | (out[z_names].max(axis=1, skipna=True) >= bad_z)
    )
    out["is_bad_trial"] = hard_bad

    reasons: list[str] = []
    reason_map = {
        "nan_fraction": "NaN/Inf",
        "flat_channel_fraction": "flat_channels",
        "z_p2p": "large_p2p",
        "z_max_p2p": "extreme_channel_p2p",
        "z_rms": "large_rms",
        "z_max_abs": "large_absolute_amplitude",
        "z_jump": "jump/electrode_pop",
        "z_common_mode": "global_movement/common_mode",
        "z_kurtosis": "non_gaussian_spike",
        "z_low_freq": "slow_drift",
        "z_high_freq": "muscle_high_frequency",
        "z_line_noise": "line_noise_60Hz",
    }

    for _, row in out.iterrows():
        row_reasons: list[str] = []
        if row["nan_fraction"] > 0:
            row_reasons.append(reason_map["nan_fraction"])
        if row["flat_channel_fraction"] >= 0.10:
            row_reasons.append(reason_map["flat_channel_fraction"])
        for col in z_names:
            if np.isfinite(row[col]) and row[col] >= bad_z:
                row_reasons.append(reason_map[col])
        reasons.append(";".join(row_reasons))

    out["bad_reasons"] = reasons
    return out


def compute_channel_metrics(X_uv: np.ndarray) -> pd.DataFrame:
    finite_X = np.where(np.isfinite(X_uv), X_uv, np.nan)
    # trial/time 전체를 합친 채널별 특성
    channel_std = np.nanstd(finite_X, axis=(0, 2))
    channel_rms = np.sqrt(np.nanmean(finite_X**2, axis=(0, 2)))
    channel_p2p_median = np.nanmedian(
        np.nanmax(finite_X, axis=2) - np.nanmin(finite_X, axis=2),
        axis=0,
    )
    channel_nan_fraction = np.mean(~np.isfinite(X_uv), axis=(0, 2))

    df = pd.DataFrame(
        {
            "channel_index": np.arange(X_uv.shape[1]),
            "std_uv": channel_std,
            "rms_uv": channel_rms,
            "median_trial_p2p_uv": channel_p2p_median,
            "nan_fraction": channel_nan_fraction,
        }
    )

    for col in ("std_uv", "rms_uv", "median_trial_p2p_uv"):
        df[f"z_{col}"] = robust_z(np.log1p(np.maximum(df[col].to_numpy(float), 0)))

    df["is_suspicious_channel"] = (
        (df["nan_fraction"] > 0)
        | (df["z_std_uv"].abs() >= 4.0)
        | (df["z_rms_uv"].abs() >= 4.0)
        | (df["z_median_trial_p2p_uv"].abs() >= 4.0)
    )
    return df


def plot_bar(
    values: pd.Series,
    title: str,
    ylabel: str,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(values)), 4.8))
    values.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_hist(
    values: pd.Series,
    title: str,
    xlabel: str,
    output: Path,
) -> None:
    finite = values[np.isfinite(values)]
    if finite.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.hist(finite, bins=min(40, max(10, int(math.sqrt(len(finite))))))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Trials")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    output: Path,
    normalize: bool,
    title: str,
) -> None:
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        normalize="true" if normalize else None,
    )

    fig_size = max(6, 0.7 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(cm, interpolation="nearest", aspect="auto")
    fig.colorbar(image, ax=ax)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted phoneme",
        ylabel="True phoneme",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    threshold = np.nanmax(cm) / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def prediction_frame_from_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"subject", "y_true", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    df["subject"] = df["subject"].astype(str).str.upper()
    df["y_true"] = decode_strings(df["y_true"].to_numpy())
    df["y_pred"] = decode_strings(df["y_pred"].to_numpy())
    return df


def prediction_frame_from_npz(path: Path, subject_override: str | None = None) -> pd.DataFrame:
    with np.load(path, allow_pickle=True) as npz:
        true_key = first_existing_key(npz, None, PRED_TRUE_CANDIDATES)
        pred_key = first_existing_key(npz, None, PRED_PRED_CANDIDATES)
        subject_key = first_existing_key(
            npz,
            None,
            SUBJECT_KEY_CANDIDATES,
            required=False,
        )
        prob_key = first_existing_key(
            npz,
            None,
            PRED_PROB_CANDIDATES,
            required=False,
        )

        y_true = decode_strings(npz[true_key])
        y_pred = decode_strings(npz[pred_key])

        if len(y_true) != len(y_pred):
            raise ValueError(
                f"{path}: len(y_true)={len(y_true)} != len(y_pred)={len(y_pred)}"
            )

        if subject_key is not None:
            subjects = decode_strings(npz[subject_key])
            if len(subjects) == 1:
                subjects = np.repeat(subjects[0], len(y_true))
        elif subject_override is not None:
            subjects = np.repeat(subject_override, len(y_true))
        else:
            inferred = infer_subject(path)
            if inferred is None:
                raise ValueError(
                    f"{path}: no subject array and subject could not be inferred."
                )
            subjects = np.repeat(inferred, len(y_true))

        if len(subjects) != len(y_true):
            raise ValueError(
                f"{path}: subject length {len(subjects)} != predictions {len(y_true)}"
            )

        df = pd.DataFrame(
            {
                "subject": pd.Series(subjects).astype(str).str.upper(),
                "y_true": y_true,
                "y_pred": y_pred,
            }
        )

        if prob_key is not None:
            probs = np.asarray(npz[prob_key])
            if probs.ndim == 2 and probs.shape[0] == len(df):
                clipped = np.clip(probs.astype(float), 1e-12, 1.0)
                df["prediction_confidence"] = np.nanmax(clipped, axis=1)
                df["prediction_entropy"] = -np.nansum(
                    clipped * np.log(clipped),
                    axis=1,
                )
        return df


def load_predictions(args: argparse.Namespace) -> pd.DataFrame | None:
    frames: list[pd.DataFrame] = []

    if args.pred_csv:
        frames.append(prediction_frame_from_csv(args.pred_csv))

    if args.pred_npz:
        frames.append(prediction_frame_from_npz(args.pred_npz))

    if args.pred_dir:
        for subject in args.subjects:
            candidates = sorted(args.pred_dir.rglob(f"*{subject}*.npz"))
            if not candidates:
                warnings.warn(f"[{subject}] no prediction NPZ found in {args.pred_dir}")
                continue
            frames.append(
                prediction_frame_from_npz(candidates[0], subject_override=subject)
            )

    if not frames:
        return None

    pred = pd.concat(frames, ignore_index=True)
    pred = pred[pred["subject"].isin([s.upper() for s in args.subjects])].copy()
    return pred


def evaluate_predictions(
    subject: str,
    pred_df: pd.DataFrame,
    trial_metrics: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    sub = pred_df[pred_df["subject"] == subject].copy()
    if sub.empty:
        return {
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "macro_f1": np.nan,
            "n_prediction_trials": 0,
        }

    # 빈 라벨 제거
    valid = (
        sub["y_true"].astype(str).str.len().gt(0)
        & sub["y_pred"].astype(str).str.len().gt(0)
    )
    sub = sub[valid].copy()

    y_true = sub["y_true"].to_numpy(dtype=object)
    y_pred = sub["y_pred"].to_numpy(dtype=object)
    labels = sorted(set(y_true) | set(y_pred))

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "n_prediction_trials": len(sub),
        "accuracy_clean_trials": np.nan,
        "accuracy_flagged_trials": np.nan,
        "n_clean_prediction_trials": 0,
        "n_flagged_prediction_trials": 0,
        "accuracy_gain_clean_vs_all": np.nan,
    }

    # Prediction 행과 EEG trial 행의 순서가 같을 때 잡음 trial 제거 효과를 직접 계산.
    # 길이가 다르면 잘못된 대응을 피하기 위해 계산하지 않습니다.
    if len(sub) == len(trial_metrics):
        sub = sub.reset_index(drop=True)
        aligned_quality = trial_metrics.reset_index(drop=True)
        sub["is_bad_trial"] = aligned_quality["is_bad_trial"].to_numpy(bool)
        sub["bad_score"] = aligned_quality["bad_score"].to_numpy(float)
        sub["eeg_label"] = aligned_quality["label"].astype(str).to_numpy()
        sub["trial_id"] = aligned_quality["trial_id"].astype(str).to_numpy()
        sub["label_matches_eeg_data"] = (
            sub["y_true"].astype(str).to_numpy()
            == sub["eeg_label"].astype(str).to_numpy()
        )
        sub.to_csv(output_dir / "predictions_with_eeg_quality.csv", index=False)

        clean = ~sub["is_bad_trial"].to_numpy(bool)
        flagged = ~clean
        metrics["n_clean_prediction_trials"] = int(clean.sum())
        metrics["n_flagged_prediction_trials"] = int(flagged.sum())
        metrics["prediction_eeg_label_match_fraction"] = float(
            sub["label_matches_eeg_data"].mean()
        )

        if clean.sum() > 0:
            clean_acc = accuracy_score(
                sub.loc[clean, "y_true"],
                sub.loc[clean, "y_pred"],
            )
            metrics["accuracy_clean_trials"] = float(clean_acc)
            metrics["accuracy_gain_clean_vs_all"] = float(
                clean_acc - metrics["accuracy"]
            )
        if flagged.sum() > 0:
            metrics["accuracy_flagged_trials"] = float(
                accuracy_score(
                    sub.loc[flagged, "y_true"],
                    sub.loc[flagged, "y_pred"],
                )
            )

        quality_accuracy = (
            sub.groupby("is_bad_trial")
            .apply(lambda g: pd.Series({
                "n_trials": len(g),
                "accuracy": float((g["y_true"] == g["y_pred"]).mean()),
                "mean_bad_score": float(g["bad_score"].mean()),
            }), include_groups=False)
            .reset_index()
        )
        quality_accuracy.to_csv(
            output_dir / "accuracy_clean_vs_flagged_trials.csv",
            index=False,
        )

    plot_confusion(
        y_true,
        y_pred,
        labels,
        output_dir / "confusion_matrix_counts.png",
        normalize=False,
        title=f"{subject} confusion matrix (counts)",
    )
    plot_confusion(
        y_true,
        y_pred,
        labels,
        output_dir / "confusion_matrix_normalized.png",
        normalize=True,
        title=f"{subject} confusion matrix (row-normalized)",
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        output_dir / "confusion_matrix_counts.csv"
    )
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    pd.DataFrame(cm_norm, index=labels, columns=labels).to_csv(
        output_dir / "confusion_matrix_normalized.csv"
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(output_dir / "classification_report.csv")

    # 어떤 실제 음소가 어떤 음소로 자주 잘못 예측되는지 정렬.
    errors = sub[sub["y_true"] != sub["y_pred"]]
    if not errors.empty:
        error_pairs = (
            errors.groupby(["y_true", "y_pred"])
            .size()
            .rename("count")
            .reset_index()
            .sort_values("count", ascending=False)
        )
        error_pairs.to_csv(output_dir / "top_confusion_pairs.csv", index=False)

    return metrics


def compare_prediction_labels_to_data(
    subject: str,
    data_labels: np.ndarray,
    pred_df: pd.DataFrame | None,
) -> dict[str, Any]:
    if pred_df is None:
        return {}

    sub = pred_df[pred_df["subject"] == subject]
    if sub.empty:
        return {"prediction_label_set_mismatch": np.nan}

    data_set = set(data_labels[data_labels != ""])
    pred_true_set = set(sub["y_true"].astype(str)) - {""}

    return {
        "prediction_label_set_mismatch": data_set != pred_true_set,
        "classes_only_in_data": ";".join(sorted(data_set - pred_true_set)),
        "classes_only_in_predictions": ";".join(sorted(pred_true_set - data_set)),
        "n_data_trials_minus_prediction_trials": len(data_labels) - len(sub),
    }


def save_subject_plots(
    subject: str,
    labels: np.ndarray,
    trial_metrics: pd.DataFrame,
    output_dir: Path,
) -> None:
    counts = pd.Series(labels[labels != ""]).value_counts().sort_index()
    plot_bar(
        counts,
        f"{subject} phoneme trial counts",
        "Trials",
        output_dir / "phoneme_trial_counts.png",
    )

    for col, label in (
        ("p2p_uv", "Median channel peak-to-peak (uV)"),
        ("rms_uv", "Median channel RMS (uV)"),
        ("jump_ratio", "Jump ratio"),
        ("common_mode_ratio", "Common-mode ratio"),
        ("low_freq_ratio", "Low-frequency ratio"),
        ("high_freq_ratio", "High-frequency ratio"),
        ("line_noise_ratio", "59-61 Hz ratio"),
        ("median_kurtosis", "Median channel kurtosis"),
        ("bad_score", "Bad-trial score"),
    ):
        plot_hist(
            trial_metrics[col],
            f"{subject} {label}",
            label,
            output_dir / f"distribution_{col}.png",
        )

    bad_rate_by_label = (
        trial_metrics.assign(label=labels)
        .query("label != ''")
        .groupby("label")["is_bad_trial"]
        .agg(["sum", "count", "mean"])
        .rename(
            columns={
                "sum": "bad_trials",
                "count": "total_trials",
                "mean": "bad_fraction",
            }
        )
        .sort_index()
    )
    bad_rate_by_label.to_csv(output_dir / "bad_trial_rate_by_phoneme.csv")

    if not bad_rate_by_label.empty:
        plot_bar(
            bad_rate_by_label["bad_fraction"],
            f"{subject} bad-trial fraction by phoneme",
            "Bad-trial fraction",
            output_dir / "bad_trial_fraction_by_phoneme.png",
        )


def process_subject(
    subject: str,
    path: Path,
    args: argparse.Namespace,
    pred_df: pd.DataFrame | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    subject_dir = args.output_dir / subject
    subject_dir.mkdir(parents=True, exist_ok=True)

    with np.load(path, allow_pickle=True) as npz:
        x_key = first_existing_key(npz, args.x_key, X_CANDIDATES)
        label_key = first_existing_key(npz, args.label_key, LABEL_CANDIDATES)
        labels_all = decode_strings(npz[label_key])
        X_all = orient_x_to_trials(np.asarray(npz[x_key]), len(labels_all))

        combined_subject_mask = None
        if args.combined_data_npz is not None:
            subject_key = first_existing_key(
                npz,
                None,
                SUBJECT_KEY_CANDIDATES,
                required=True,
            )
            subjects_all = decode_strings(npz[subject_key])
            if len(subjects_all) == 1:
                subjects_all = np.repeat(subjects_all[0], len(labels_all))
            if len(subjects_all) != len(labels_all):
                raise ValueError(
                    f"[{subject}] subject array length {len(subjects_all)} "
                    f"!= labels {len(labels_all)}"
                )
            combined_subject_mask = np.char.upper(
                subjects_all.astype(str)
            ) == subject.upper()
            if not np.any(combined_subject_mask):
                raise ValueError(
                    f"[{subject}] is absent from subject_per_trial in {path}"
                )
            labels = labels_all[combined_subject_mask]
            X = X_all[combined_subject_mask]
        else:
            labels = labels_all
            X = X_all

        sfreq_key = first_existing_key(
            npz,
            None,
            SFREQ_CANDIDATES,
            required=False,
        )
        sfreq = args.sfreq if args.sfreq is not None else get_scalar(npz, sfreq_key)
        if sfreq is None:
            raise ValueError(
                f"[{subject}] Sampling frequency is missing. Use --sfreq 256."
            )

        trial_id_key = first_existing_key(
            npz,
            args.trial_id_key,
            TRIAL_ID_CANDIDATES,
            required=False,
        )
        if trial_id_key is not None:
            trial_ids_all = decode_strings(npz[trial_id_key])
            if combined_subject_mask is not None and len(trial_ids_all) == len(labels_all):
                trial_ids = trial_ids_all[combined_subject_mask]
            else:
                trial_ids = trial_ids_all
            if len(trial_ids) != len(labels):
                warnings.warn(
                    f"[{subject}] trial ID length {len(trial_ids)} "
                    f"!= labels {len(labels)}; using array index."
                )
                trial_ids = np.asarray([str(i) for i in range(len(labels))])
        else:
            trial_ids = np.asarray([str(i) for i in range(len(labels))])

        ch_key = first_existing_key(
            npz,
            None,
            CHANNEL_NAME_CANDIDATES,
            required=False,
        )
        if ch_key is not None:
            ch_names = decode_strings(npz[ch_key])
            if len(ch_names) != X.shape[1]:
                ch_names = np.asarray([f"CH{i:02d}" for i in range(X.shape[1])])
        else:
            ch_names = np.asarray([f"CH{i:02d}" for i in range(X.shape[1])])

        available_keys = list(npz.files)

    X = X.astype(np.float64, copy=False)
    X_uv, unit_note = to_microvolts(X, args.unit)

    trial_metrics = compute_trial_metrics(
        X_uv,
        sfreq=sfreq,
        max_welch_trials=args.max_welch_trials,
    )
    trial_metrics = add_bad_trial_scores(trial_metrics, bad_z=args.bad_z)
    trial_metrics.insert(0, "subject", subject)
    trial_metrics.insert(1, "trial_id", trial_ids)
    trial_metrics.insert(2, "label", labels)
    trial_metrics.to_csv(subject_dir / "trial_quality.csv", index=False)

    bad_trials = trial_metrics[trial_metrics["is_bad_trial"]].sort_values(
        ["bad_score", "max_abs_uv"],
        ascending=False,
    )
    bad_trials.to_csv(subject_dir / "bad_trials_ranked.csv", index=False)

    channel_metrics = compute_channel_metrics(X_uv)
    channel_metrics.insert(0, "subject", subject)
    channel_metrics.insert(2, "channel_name", ch_names)
    channel_metrics.to_csv(subject_dir / "channel_quality.csv", index=False)
    channel_metrics[channel_metrics["is_suspicious_channel"]].to_csv(
        subject_dir / "suspicious_channels.csv",
        index=False,
    )

    label_counts = pd.Series(labels[labels != ""]).value_counts().sort_index()
    label_counts.rename("trial_count").to_csv(subject_dir / "phoneme_trial_counts.csv")

    # 중복 trial ID와 충돌 라벨 확인
    ids_df = pd.DataFrame({"trial_id": trial_ids, "label": labels})
    duplicate_ids = ids_df[ids_df.duplicated("trial_id", keep=False)].copy()
    duplicate_ids.to_csv(subject_dir / "duplicate_trial_ids.csv", index=False)

    conflicting = (
        ids_df.groupby("trial_id")["label"]
        .nunique()
        .loc[lambda s: s > 1]
        .index
    )
    conflicting_df = ids_df[ids_df["trial_id"].isin(conflicting)].sort_values(
        ["trial_id", "label"]
    )
    conflicting_df.to_csv(
        subject_dir / "conflicting_labels_for_same_trial_id.csv",
        index=False,
    )

    save_subject_plots(subject, labels, trial_metrics, subject_dir)

    prediction_metrics = {}
    if pred_df is not None:
        prediction_metrics = evaluate_predictions(
            subject,
            pred_df,
            trial_metrics,
            subject_dir,
        )

    label_prediction_check = compare_prediction_labels_to_data(
        subject,
        labels,
        pred_df,
    )

    valid_counts = label_counts[label_counts > 0]
    imbalance_ratio = (
        float(valid_counts.max() / valid_counts.min())
        if len(valid_counts) >= 2 and valid_counts.min() > 0
        else np.nan
    )

    summary: dict[str, Any] = {
        "subject": subject,
        "data_file": str(path),
        "x_key": x_key,
        "label_key": label_key,
        "available_npz_keys": ";".join(available_keys),
        "shape_trials": X.shape[0],
        "shape_channels": X.shape[1],
        "shape_timepoints": X.shape[2],
        "sfreq": sfreq,
        "unit_interpretation": unit_note,
        "n_empty_labels": int(np.sum(labels == "")),
        "n_classes": int(len(label_counts)),
        "min_trials_per_class": int(label_counts.min()) if len(label_counts) else 0,
        "max_trials_per_class": int(label_counts.max()) if len(label_counts) else 0,
        "class_imbalance_ratio": imbalance_ratio,
        "n_duplicate_trial_rows": int(len(duplicate_ids)),
        "n_conflicting_trial_ids": int(len(conflicting)),
        "bad_trial_fraction": float(trial_metrics["is_bad_trial"].mean()),
        "n_bad_trials": int(trial_metrics["is_bad_trial"].sum()),
        "n_suspicious_channels": int(
            channel_metrics["is_suspicious_channel"].sum()
        ),
        "median_p2p_uv": float(trial_metrics["p2p_uv"].median()),
        "median_rms_uv": float(trial_metrics["rms_uv"].median()),
        "median_jump_ratio": float(trial_metrics["jump_ratio"].median()),
        "median_common_mode_ratio": float(
            trial_metrics["common_mode_ratio"].median()
        ),
        "median_low_freq_ratio": float(
            trial_metrics["low_freq_ratio"].median()
        ),
        "median_high_freq_ratio": float(
            trial_metrics["high_freq_ratio"].median()
        ),
        "median_line_noise_ratio": float(
            trial_metrics["line_noise_ratio"].median()
        ),
        "median_kurtosis": float(trial_metrics["median_kurtosis"].median()),
    }
    summary.update(prediction_metrics)
    summary.update(label_prediction_check)

    return summary, label_counts.rename(subject)


def add_subject_outlier_flags(
    summary_df: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    out = summary_df.copy()

    metric_cols = [
        "bad_trial_fraction",
        "n_suspicious_channels",
        "median_p2p_uv",
        "median_rms_uv",
        "median_jump_ratio",
        "median_common_mode_ratio",
        "median_low_freq_ratio",
        "median_high_freq_ratio",
        "median_line_noise_ratio",
        "median_kurtosis",
        "class_imbalance_ratio",
        "n_empty_labels",
        "n_conflicting_trial_ids",
    ]

    flag_cols: list[str] = []
    for col in metric_cols:
        if col not in out:
            continue
        values = pd.to_numeric(out[col], errors="coerce").to_numpy(float)
        if np.all(values[np.isfinite(values)] >= 0):
            z = robust_z(np.log1p(values))
        else:
            z = robust_z(values)
        z_col = f"cohort_z_{col}"
        flag_col = f"outlier_{col}"
        out[z_col] = z
        out[flag_col] = z >= threshold
        flag_cols.append(flag_col)

    out["n_subject_level_outlier_metrics"] = out[flag_cols].sum(axis=1)
    out["subject_data_outlier"] = out["n_subject_level_outlier_metrics"] > 0
    return out


def write_interpretation(
    summary_df: pd.DataFrame,
    focus_subjects: list[str],
    output: Path,
) -> None:
    lines: list[str] = []
    lines.append("EEG SUBJECT AUDIT - 자동 해석 보조")
    lines.append("=" * 72)
    lines.append("")
    lines.append(
        "주의: 이 결과는 '진단'이 아니라 데이터 문제 후보를 좁히기 위한 자동 screening입니다."
    )
    lines.append("")

    for subject in focus_subjects:
        row_df = summary_df[summary_df["subject"] == subject]
        if row_df.empty:
            lines.append(f"[{subject}] 데이터 없음")
            lines.append("")
            continue

        row = row_df.iloc[0]
        lines.append(f"[{subject}]")
        lines.append(
            f"- accuracy={row.get('accuracy', np.nan):.4f}, "
            f"balanced_accuracy={row.get('balanced_accuracy', np.nan):.4f}, "
            f"macro_F1={row.get('macro_f1', np.nan):.4f}"
        )
        if np.isfinite(row.get("accuracy_clean_trials", np.nan)):
            lines.append(
                f"- clean-trial accuracy={row.get('accuracy_clean_trials', np.nan):.4f}, "
                f"flagged-trial accuracy={row.get('accuracy_flagged_trials', np.nan):.4f}, "
                f"clean vs all gain={row.get('accuracy_gain_clean_vs_all', np.nan):+.4f}"
            )
        lines.append(
            f"- bad trials={int(row.get('n_bad_trials', 0))}/"
            f"{int(row.get('shape_trials', 0))} "
            f"({100 * float(row.get('bad_trial_fraction', np.nan)):.1f}%)"
        )
        lines.append(
            f"- suspicious channels={int(row.get('n_suspicious_channels', 0))}, "
            f"empty labels={int(row.get('n_empty_labels', 0))}, "
            f"conflicting trial IDs={int(row.get('n_conflicting_trial_ids', 0))}"
        )
        lines.append(
            f"- class imbalance ratio={row.get('class_imbalance_ratio', np.nan):.3f}, "
            f"subject-level outlier metrics="
            f"{int(row.get('n_subject_level_outlier_metrics', 0))}"
        )

        suggestions: list[str] = []
        if bool(row.get("prediction_label_set_mismatch", False)):
            suggestions.append(
                "data label set과 prediction y_true label set이 다름: split/label encoding 확인"
            )
        if row.get("n_data_trials_minus_prediction_trials", 0) != 0:
            suggestions.append(
                "원본 trial 수와 prediction trial 수가 다름: filtering/order mismatch 확인"
            )
        if row.get("n_empty_labels", 0) > 0:
            suggestions.append("빈 phoneme label 존재")
        if row.get("n_conflicting_trial_ids", 0) > 0:
            suggestions.append("같은 trial ID에 서로 다른 label 존재")
        if row.get("class_imbalance_ratio", 1) >= 2:
            suggestions.append("phoneme별 trial 수 불균형 큼")
        if row.get("bad_trial_fraction", 0) >= 0.10:
            suggestions.append("잡음/이상 trial 비율 높음")
        if row.get("accuracy_gain_clean_vs_all", 0) >= 0.05:
            suggestions.append(
                "이상 trial 제외 후 정확도가 5%p 이상 회복: 데이터 잡음이 성능 저하에 기여"
            )
        if row.get("n_suspicious_channels", 0) >= 2:
            suggestions.append("반복적으로 이상한 채널 존재")
        if row.get("cohort_z_median_common_mode_ratio", 0) >= 3.5:
            suggestions.append("전체 채널 동시 변동이 커 movement/global artifact 의심")
        if row.get("cohort_z_median_high_freq_ratio", 0) >= 3.5:
            suggestions.append("30-50 Hz 비율이 커 근전도/움직임 잡음 의심")
        if row.get("cohort_z_median_low_freq_ratio", 0) >= 3.5:
            suggestions.append("저주파 drift가 다른 피험자보다 큼")
        if row.get("cohort_z_median_line_noise_ratio", 0) >= 3.5:
            suggestions.append("59-61 Hz 잔류가 큼: notch/filter 적용 확인")
        if row.get("cohort_z_median_jump_ratio", 0) >= 3.5:
            suggestions.append("급격한 jump/electrode pop이 큼")

        if suggestions:
            lines.append("- 확인 우선순위:")
            for s in suggestions:
                lines.append(f"  * {s}")
        else:
            lines.append(
                "- 뚜렷한 라벨/데이터 품질 outlier가 없으면 모델이 해당 피험자의 "
                "개인차를 일반화하지 못한 가능성이 더 큼"
            )
        lines.append("")

    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.subjects = [s.upper() for s in args.subjects]
    args.focus_subjects = [s.upper() for s in args.focus_subjects]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.combined_data_npz is not None:
        if not args.combined_data_npz.exists():
            print(
                f"Combined NPZ does not exist: {args.combined_data_npz}",
                file=sys.stderr,
            )
            return 2
        subject_files = {s: args.combined_data_npz for s in args.subjects}
    else:
        subject_files = find_subject_files(
            args.data_dir,
            args.subjects,
            args.file_pattern,
        )
        missing_subjects = [s for s in args.subjects if s not in subject_files]
        if missing_subjects:
            warnings.warn(f"No EEG NPZ found for: {missing_subjects}")

        if not subject_files:
            print("No subject NPZ files found.", file=sys.stderr)
            return 2

    pred_df = load_predictions(args)
    if pred_df is not None:
        pred_df.to_csv(args.output_dir / "combined_predictions_loaded.csv", index=False)

    summaries: list[dict[str, Any]] = []
    count_series: list[pd.Series] = []
    failures: list[dict[str, str]] = []

    for subject in args.subjects:
        if subject not in subject_files:
            continue
        path = subject_files[subject]
        print(f"\n[{subject}] Processing {path}")
        try:
            summary, counts = process_subject(
                subject,
                path,
                args,
                pred_df,
            )
            summaries.append(summary)
            count_series.append(counts)
            print(
                f"[{subject}] done: trials={summary['shape_trials']}, "
                f"classes={summary['n_classes']}, "
                f"bad={100 * summary['bad_trial_fraction']:.1f}%"
            )
        except Exception as exc:
            failures.append(
                {
                    "subject": subject,
                    "file": str(path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(f"[{subject}] FAILED: {type(exc).__name__}: {exc}")

    if failures:
        pd.DataFrame(failures).to_csv(
            args.output_dir / "failures.csv",
            index=False,
        )

    if not summaries:
        print("All subjects failed. See failures.csv.", file=sys.stderr)
        return 1

    summary_df = pd.DataFrame(summaries)
    summary_df = add_subject_outlier_flags(
        summary_df,
        threshold=args.subject_outlier_z,
    )
    summary_df.to_csv(args.output_dir / "subject_summary.csv", index=False)

    if count_series:
        counts_df = pd.concat(count_series, axis=1).fillna(0).astype(int)
        counts_df.index.name = "phoneme"
        counts_df.to_csv(args.output_dir / "phoneme_counts_all_subjects.csv")

        # 각 phoneme에서 피험자별 trial 수 cohort outlier 탐색
        rows = []
        for phoneme, row in counts_df.iterrows():
            values = row.to_numpy(float)
            z = robust_z(values)
            for subject, count, z_value in zip(row.index, values, z):
                rows.append(
                    {
                        "subject": subject,
                        "phoneme": phoneme,
                        "trial_count": int(count),
                        "cohort_z_trial_count": z_value,
                        "count_outlier": bool(
                            np.isfinite(z_value)
                            and abs(z_value) >= args.subject_outlier_z
                        ),
                    }
                )
        count_outliers = pd.DataFrame(rows)
        count_outliers.to_csv(
            args.output_dir / "phoneme_count_outliers.csv",
            index=False,
        )

    # focus subject만 보기 쉬운 별도 파일
    focus_df = summary_df[
        summary_df["subject"].isin(args.focus_subjects)
    ].copy()
    focus_df.to_csv(
        args.output_dir / "focus_S06_S08_S10_summary.csv",
        index=False,
    )

    write_interpretation(
        summary_df,
        args.focus_subjects,
        args.output_dir / "READ_ME_FIRST.txt",
    )

    config = vars(args).copy()
    for key, value in config.items():
        if isinstance(value, Path):
            config[key] = str(value)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nCompleted.")
    print(f"Open first: {args.output_dir / 'READ_ME_FIRST.txt'}")
    print(f"Subject summary: {args.output_dir / 'subject_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
