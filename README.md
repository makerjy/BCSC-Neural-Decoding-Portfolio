# EEG 기반 신경 음성 디코딩

**Neural Speech Decoding for Brain-Computer Interfaces**는 2026년 6월부터 8월까지 진행된 Brain and Cognitive Society(BCSC) 10기 팀 프로젝트입니다. 이 저장소는 프로젝트의 전체 코드와 결과물을 포트폴리오 형태로 정리한 아카이브입니다.

본 프로젝트는 음성 청취 과제 중 기록된 비침습 EEG에서 음소 정보를 디코딩할 수 있는지 탐구했습니다. EEGLAB 기반 전처리 파이프라인, 자극 시점에 정렬한 EEG 특성 분석, 피험자 정보를 고려한 11개 음소 분류 모델을 함께 다룹니다.

이 저장소에서는 프로젝트 전체 결과물과 함께 원시 EEG 전처리 파이프라인, 모델링 아이디어, 최종 포스터 **Figure 3**의 분석 및 시각화 작업을 중심으로 정리했습니다.

## 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 데이터셋 | OpenNeuro [`ds006104`, version 1.0.0](https://openneuro.org/datasets/ds006104/versions/1.0.0) |
| 사용 대상 | 2021년 코호트의 참가자 16명 |
| EEG 기록 | 64채널, 2,000 Hz로 측정했으며 보조 채널 제거 후 61개 두피 채널 사용 |
| 과제 | 단일 음소 듣기·따라 말하기, 자음-모음 쌍 식별, 실제 단어·가짜 단어 분류, TMS 조건 |
| 모델 입력 | `trials × 61 channels × 256 samples` (0–1초, 256 Hz) |
| 주요 분류 과제 | 11개 음소: `a, i, e, u, o, s, z, p, b, t, d` |
| 우연 수준 | 11개 클래스 분류 기준 0.091 |

## 주요 작업

### 1. 원시 EEG 전처리 파이프라인

원시 EEGLAB 기록을 모델 학습에 사용할 수 있는 배열로 변환하는 전처리 파이프라인을 구성했습니다. 저장소에 포함된 구현의 주요 단계는 다음과 같습니다.

1. 실험 테이블을 바탕으로 구조화된 event 정보 재구성
2. 보조 채널(`EOG`, `M1`, `M2`, `BIP1–BIP24`) 제거 및 채널 위치 검증
3. 각 TMS pulse 전후 **−5 ms부터 +25 ms** 구간의 artifact interpolation
4. **256 Hz** resampling, **59–61 Hz notch filter**, **0.1–100 Hz band-pass filter** 적용
5. `stim_1`을 기준으로 **0–1초** 구간 epoch 생성
6. amplitude, peak-to-peak, RMS, sample-step 지표와 피험자별 robust threshold를 이용한 bad epoch 자동 탐지
7. ICA와 ICLabel을 사용하고 보수적인 **0.90 probability threshold**를 적용한 artifact 제거
8. task·condition·category 분리 및 `.set → .mat → .npz` 변환

전체 전처리 흐름은 [`preprocessing_matlab/preprocessing_main.m`](preprocessing_matlab/preprocessing_main.m)에서 시작합니다. 각 단계의 검증과 데이터 변환 과정을 확인할 수 있도록 관련 함수도 같은 폴더에 보존했습니다.

### 2. 모델링 아이디어

모델링 단계에서는 각 trial의 **공간적 관계**, **짧은 시간 지연 특성**, **자극에 정렬된 시간 구조**를 함께 반영하는 방향을 설계했습니다. 이를 바탕으로 구현된 모델은 각 trial을 다음과 같이 표현합니다.

```text
X: 61 channels × 256 samples
  → supervised SNR-oriented spatial projection: 61 → 14
  → A(t) = [Z(t), Z(t−20), φ0(t)…φ7(t)]
  → SPD covariance representation
  → log-Euclidean tangent-space projection
  → subject-specific Gaussian decision rule
  → 11개 음소 중 하나로 분류
```

핵심은 서로 보완적인 세 가지 정보인 **공간 covariance**, **짧은 시간 지연 특성**(20 samples, 약 78 ms), **DCT basis를 이용한 자극 정렬 시간 구조**를 함께 보존하는 것입니다. 저장소에서는 해석 가능한 설계 모델과 가장 높은 성능을 기록한 concatenated ensemble을 구분합니다. 자세한 내용은 [`model/model/README.md`](model/model/README.md)와 [`model/SUMMARY.md`](model/SUMMARY.md)에서 확인할 수 있습니다.

### 3. 포스터 Figure 3

연속 EEG를 자극 시점에 맞춰 정렬한 뒤 ERP, 자극 전후 RMS, scalp topography를 분석하고, 그 결과를 최종 포스터 Figure 3으로 시각화했습니다.

- **3A — Continuous EEG와 event alignment:** 단일 trial EEG가 `stim_1 = 0 s`에 정확히 동기화되는지 확인했습니다.
- **3B — 평균 ERP:** 정렬된 trial을 평균하여 작은 N100-like negative deflection과 **400–500 ms** 부근의 late positive response를 확인했습니다.
- **3C — 자극 전후 RMS:** 자극 전 **−0.5~−0.1초**와 자극 후 **0.1~0.5초**를 비교했으며, frontal electrode에서 가장 큰 증가가 나타났습니다.
- **3D — ERP scalp topography:** **100–500 ms** 동안 자극 관련 활동이 두피 전반에서 재분포되는 양상을 시각화했습니다.

관련 코드는 [`analysis/`](analysis/)에 있으며, 주요 스크립트는 [`figure1.py`](analysis/figure1.py), [`average_stage3_erp.py`](analysis/average_stage3_erp.py), [`plot_stage3_erp_topomaps.py`](analysis/plot_stage3_erp_topomaps.py)입니다.

## 모델링 결과

아래는 팀에서 함께 수행한 모델링의 최종 평가 결과입니다.

| 모델 및 평가 방식 | 11개 음소 balanced accuracy |
|---|---:|
| 설계한 main model, 미사용 최종 테스트 세트 | 0.245 |
| 최고 성능의 concatenated representation, 미사용 최종 테스트 세트 | **0.306** |
| 우연 수준 | 0.091 |
| 최고 성능 모델, leave-one-subject-out | 0.143 |

이 결과를 해석할 때는 피험자 내 calibration과 새로운 피험자에 대한 일반화 성능의 차이를 고려해야 합니다. 0.306은 calibration data가 제공된 피험자의 새로운 trial에 대한 결과이며, 완전히 새로운 피험자에 대한 성능으로 제시해서는 안 됩니다. 세부 대조 실험(control experiment), null test, 한계점은 [`model/SUMMARY.md`](model/SUMMARY.md)에 정리되어 있습니다.

## 최종 포스터

[2페이지 분량의 원본 포스터 내려받기 (PDF)](assets/poster/BCSC_Neural_Decoding_Poster.pdf)

![포스터 1페이지](assets/poster/poster-page-1.png)

![포스터 2페이지](assets/poster/poster-page-2.png)

## 저장소 구조

```text
.
├── preprocessing_matlab/   # 원시 EEG 전처리, epoch 생성, ICA 검토
├── analysis/               # QC, event alignment, ERP/RMS, scalp topography
├── model/
│   ├── model/              # 주요 모델과 공통 구현
│   ├── attic/              # 탐색 실험 및 control experiment
│   ├── logs/               # 보존된 실험 로그
│   └── slides/             # 팀 모델링 발표 자료와 미리보기
├── assets/poster/          # 포스터 PDF와 렌더링 이미지
├── CONTRIBUTIONS.md        # 프로젝트 작업 범위와 기여 내용
├── REPRODUCIBILITY.md      # 실행 환경, 데이터, 재현 방법
└── README.original.md      # 원본 팀 저장소 README
```

## 실행 및 재현

개인정보 보호와 저장 용량을 고려하여 원시·전처리 EEG 데이터는 Git에 포함하지 않았습니다. Python 의존성은 다음 명령으로 설치할 수 있습니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

MATLAB 전처리를 실행하려면 EEGLAB과 ICLabel plugin이 추가로 필요합니다. 스크립트를 실행하기 전에 [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)를 확인해 주세요. 일부 보존된 실험 코드는 원본 `npz2/` 데이터 구조를 사용합니다.

## 출처와 공개 범위

최종 포스터에 기재된 팀원은 문경은(팀장), 노현이, 구민주, 김소영, 윤시원, 이재용, 조연우, 채보경입니다.

이 저장소는 BCSC 팀 프로젝트의 코드와 결과물을 포트폴리오 용도로 재구성하고 문서화한 미러 저장소입니다. 참가자의 원시 EEG 데이터는 포함하지 않았으며, 자료의 출처와 재사용 관련 안내는 [`NOTICE.md`](NOTICE.md)에서 확인할 수 있습니다.
