# 스크립트 지도

저장소 루트에 `.py`가 140개쯤 있고 대부분은 이전 세션의 탐색 흔적이다. 이 문서는 **현행**과
**대체됨**을 구분한다. 정리본은 `SUMMARY.md`, 시간순 상세 기록은 `RESULTS.md`에 있고, 1–10절이 예전 파이프라인(`npz`, 0–0.5s),
11절이 현재 파이프라인(`npz2`, 0–1.0s)이다.

## 현행 — `model/` (모델 3개 + 평가)

정리본 `SUMMARY.md`, 상세 `RESULTS.md` 11절, 모델 문서 [`model/README.md`](model/README.md).

| | 파일 | 무엇인가 | ph11 | 차원 |
|---|---|---|---|---|
① | `model/eegnet.py` | 베이스라인 EEGNet-8,2 | 0.225 | 파라미터 4,011 |
② | **`model/final_model.py`** | **메인 모델.** 통제 통과 부품만 | 0.245 | 666 |
③ | `model/best_concat.py` | 최고 성능. 표현 2개 concat, 앙상블 | **0.306** | 5406 |
| | `model/sdcore.py` | 공유 라이브러리. 세 모델이 여기서만 import하고 서로를 import하지 않는다 |
| | `model/holdout.py` | 세 모델을 미접촉 최종 test셋(시행 20%, seed 12345)에서 비교. 선택 편향 측정 |
| | `model/spdnet.py` | 통제 실험. 기하를 고정한 채 설계 vs **학습**. `SUMMARY.md` 4.3, `logs/spdnet_ph11.log` |
| | `model/reservoir.py` | 통제 실험. 역전파 없는 **창발** (리저버 27셀 스윕). `SUMMARY.md` 4.3, `logs/reservoir_*.log` |

우연 0.091. ②③은 미접촉 test셋, ①은 교차검증 15폴드.

## 통제 연구 — `attic/studies/`

`SUMMARY.md` 4절의 숫자를 만든 9개 스크립트. 모델이 아니라 "부품이 주장한 이유로 작동하는가"를
재는 실험이라 분리했다. 실행법과 대응표는 [`attic/studies/README.md`](attic/studies/README.md).
공유 라이브러리가 `model/`에 있으므로 `PYTHONPATH=model`이 필요하다.

`arch.py` (시간 기저 귀속 · 필터 2×2) · `nonlin.py` (비선형 지연) · `contrast.py` (대비별
공유도) · `budget.py` (교환비) · `geom.py` (기저 대조군) · `synth.py` (표현 비교 · LOSO) ·
`hier.py` (분할 통제) · `hier2.py` (λ/γ 표면) · `onemodel.py` (리저버 대조군)

리팩터 후 모든 스크립트가 로그의 숫자를 자릿수까지 재현하는 것을 확인했다
(`sdcore` 대 원본 함수 36개 `max|diff| = 0`, 로그 숫자 5종 정확 재현).

## `attic/` — 그 외 대체됨 (RESULTS.md 1–10절의 근거로 보존)

루트에 121개가 쌓여 있어서 주제별로 `attic/`으로 옮겼다. **삭제가 아니라 이동**이다.
1–10절의 숫자가 이 파일들에서 나왔고, 그중 14개(`shared_ceiling.py`, `stagewise.py`,
`spatial_bands.py`, `unified2.py`, `unified2b.py`, `pooled_unified.py`, `pooled_align.py`,
`pooled_factor.py`, `bayes_ard.py`, `mkl_riem.py`, `riem_kernel.py`, `align.py`, `fact.py`,
`best_model3.py`)는 **git에도 없는 유일본**이다.

| 디렉터리 | 개수 | 내용 |
|---|---|---|
`attic/pooling/` | 16 | 피험자 통합·정렬 계보. `shared_ceiling.py`의 "공유 129%"는 4.4에서 반증 |
`attic/studies/` | 10 | **통제 연구.** `SUMMARY.md` 4절을 만든 코드 — 위 절 참조 |
`attic/prepro/` | 9 | 전처리·QC (+`prepro`) |
`attic/audit/` | 7 | 누수·confound·TMS 감사 |
`attic/speed/` | 7 | 고속 LDA/Riemannian, JAX 포팅, 검증 유틸 |
`attic/bestmodel/` | 7 | 최고 모델 계보 → `model/final_model.py`가 대체 |
`attic/decode/` | 6 | 표상·융합·전이 탐색 (ROCKET, Xdawn, 필터뱅크, task 통합 등) |
`attic/dda/` | 6 | DDA 지연·카오스. a₃ 부재는 4.3에서 재확인 |
`attic/kernel/` | 4 | MKL·측도 커널. 블록 concat + LDA에 짐 |
`attic/unified/` | 4 | 통합 SPD 초기형 → 시간 기저를 추가해 `model/`이 대체 |
`attic/figs/` | 2 | 예전 ERP 그림 |
`attic/misc/` | 2 | `eda.ipynb`, `final_results.csv` |
`attic/data_ml_ready/` | 1 | 예전 파생 데이터 (0–0.5s 창) |
`attic/figures/` | 1 | 예전 그림 |

예전 데이터셋(`npz`, 992MB)과 `attic/data_ml_ready/`(20MB)는 0–0.5s 창이라 11절 숫자와
직접 비교하면 안 된다.

| 묶음 | 파일 | 11절에서 대체한 것 |
|---|---|---|
| 통합 SPD 초기형 | `unified.py`, `unified2.py`, `unified2b.py`, `pooled_unified.py` | `onemodel.py` (시간 기저 추가), `synth.py` (밴드별) |
| DDA | `dda_bank.py`, `dda_chaos.py`, `dda_delaysearch.py`, `dda_diagnostics.py`, `dda_reservoir.py`, `dda_vs_volt.py` | `nonlin.py` (모델 안에서 2차항 측정) |
| 피험자 통합 | `pooled_training.py`, `pooled_net.py`, `pooled_factor.py`, `pooled_align.py`, `hyperalign.py`, `hyperalign_check.py`, `xsubj_align.py`, `alldata_xsubj.py`, `riemann_xsubj.py`, `shared_ceiling.py`, `stagewise.py`, `bayes_ard.py`, `spatial_bands.py` | `hier.py`/`hier2.py`/`contrast.py`/`budget.py` — `shared_ceiling.py`의 "공유 129%"는 11.4에서 반증됨 |
| 커널 | `mkl_riem.py`, `mkl_test.py`, `riem_kernel.py`, `measure_kernel.py`, `conceptors.py` | 없음 (MKL 경로는 블록 concat + LDA에 짐) |
| 최고 모델 계보 | `best_model.py`, `best_model2.py`, `best_model3.py`, `mean_on_bm2.py`, `restore_mean.py`, `final.py`, `optimize_final.py` | `model/final_model.py` |
| 전처리·QC | `prepro`, `add_frequency.py`, `csd.py`, `csd_full_snr.py`, `qc2.py`, `erp_check.py`, `fetch_events.py`, `extract_pstream.py` | 유지 (npz2는 동료 전처리 산출물) |
| 누수 감사 | `leak.py`, `confound_audit.py`, `loto.py`, `syllable_audit.py`, `cvvc_diagnose.py`, `decode_confound.py`, `tms_confirm.py` | `hier.py --split token/time`, `final_model.py --study perm` |
| 속도 | `fastlda*.py`, `fastriem.py`, `jax_pipeline.py`, `run_jax.py`, `verify_*.py`, `bench.py`, `tprof.py` | 유지 (검증 유틸) |

## 삭제됨

`p2.py`, `p3.py`, `patch_pu.py`, `patch_u2.py` — 다른 `.py`를 문자열 치환으로 고쳐 쓰는
일회성 도구였다. 결과물이 전부 파일로 남아 있어서 지웠다:
`patch_u2.py` → `unified2b.py`, `patch_pu.py`→`p2.py`→`p3.py` → 현재 `pooled_unified.py`
(ProcessPoolExecutor · `PER_SUBJECT_REF=0.270` · log-Euclidean · `--workers` 모두 반영,
헤더가 `logs/pu4.log`와 일치). `dbg_lda.py` — `fastlda` 대 sklearn 수치 비교 스크래치,
같은 역할의 `verify_lda.py`가 남아 있고 git 이력에도 있다.

`__pycache__/`, `._npz2` (macOS AppleDouble)도 삭제.

**Type I 기질 시험 (`attic/type1/`, .py 4개 + .npy 22MB)** — 별개의 개인 연구이므로 이 저장소에서
제거했다. git 이력에 남아 있다.

**리저버 계보 (`attic/reservoir/`, 9개)** 와 `sdcore`의 `montage_W` / `reservoir`, `onemodel.py`의
`res_lap`/`res_rand` arm — 쓰지 않으므로 제거했다. 측정 기록은 남기되 **두 주장의 강도가 다르다는
점을 함께 남긴다**. 견고: 두피 라플라시안 W가 동일 조건 랜덤 W를 **+0.0235로 이겼다**
(15/15, 95% CI [+0.015, +0.032], p<1e-4). 약함: 지연 임베딩에 보태지 못했다 — 다만 인용되는
−0.019는 `hier2` 판독 열이고 **최종 모델이 쓰는 `persubj`에서는 −0.003, `pooled`에서는 +0.001**
이다. 단일 설정(ρ=0.90·leak=0.30·gain=1.00·tanh)에 front-end도 `basis=pca d=14 lo=20`이라
탐색이 비대칭이고, **코드를 지웠으므로 재실행할 수 없다.** 확정된 것은 "붙여도 늘지 않는다"
까지이며 "리저버가 지연 임베딩보다 못하다"는 아니다.

`.bak/`에 리팩터 전 11개 스크립트 원본이 있다. 세션의 새 파일은 git이 추적하지 않으므로
이것이 유일한 리팩터 전 사본이다. 리팩터는 함수 36개 `max|diff| = 0`과 로그 숫자 5종 정확
재현으로 검증됐으니, 커밋 후에는 지워도 된다.

## 재현 방법

```bash
# 모델 (model/)
python model/holdout.py     --task ph11                  # 세 모델 미접촉 test셋 비교
python model/final_model.py --task ph11 --study split     # 네 분할 전부
python model/final_model.py --task ph11 --study perm      # 라벨 순열 귀무
python model/final_model.py --task ph11 --study adapt     # 라벨 없는 재중심화 (LOSO)
python model/best_concat.py --task ph11 --study split
python model/best_concat.py --task ph11 --study perm
python model/eegnet.py      --task ph11 --seeds 3
python model/eegnet.py      --task ph11 --split loso      # + 라벨없는 BN 적응

# 통제 연구 (attic/studies/, 공유 라이브러리가 model/에 있어 PYTHONPATH 필요)
PYTHONPATH=model python attic/studies/arch.py     --task ph11 --study phi
PYTHONPATH=model python attic/studies/arch.py     --task ph11 --study filter
PYTHONPATH=model python attic/studies/nonlin.py   --task ph11
PYTHONPATH=model python attic/studies/contrast.py
PYTHONPATH=model python attic/studies/budget.py   --task ph11
PYTHONPATH=model python attic/studies/synth.py    --task ph11 --split loso --comb 1
```
