# attic/studies/ — 통제 연구 스크립트

`SUMMARY.md` 4절(통제 감사)의 숫자를 만든 코드다. 모델이 아니라 **"각 부품이 주장한 이유로
작동하는가"를 재는 실험**이라 `model/`에서 분리했다. 공유 라이브러리는 `model/sdcore.py`에
있으므로 실행할 때 경로를 알려줘야 한다:

```bash
PYTHONPATH=model python attic/studies/arch.py     --task ph11 --study phi     # 시간 기저 귀속
PYTHONPATH=model python attic/studies/arch.py     --task ph11 --study filter  # 필터 2×2 ablation
PYTHONPATH=model python attic/studies/nonlin.py   --task ph11                 # 비선형 지연 통제
PYTHONPATH=model python attic/studies/contrast.py                             # 대비별 공유도
PYTHONPATH=model python attic/studies/budget.py   --task ph11                 # 교환비 곡선
PYTHONPATH=model python attic/studies/geom.py     --task ph11 --basis lap     # 기저 대조군
PYTHONPATH=model python attic/studies/synth.py    --task ph11 --comb 1        # 표현 비교
PYTHONPATH=model python attic/studies/hier.py     --task ph11 --split token   # 분할 통제
PYTHONPATH=model python attic/studies/hier2.py    --task ph11                 # λ/γ 표면
PYTHONPATH=model python attic/studies/onemodel.py --task ph11                 # 통합 객체 vs 블록
```

| 파일 | 무엇을 재는가 | SUMMARY.md 절 |
|---|---|---|
`arch.py` | `--study phi` 시간 기저가 시간 순서 때문인가 (동일 행 수 통제 4종)<br>`--study filter` 필터의 피험자 억제·기하 prior가 기여하는가 (2×2) | 4.1, 4.3 |
`nonlin.py` | 비선형 지연 결합(2차항)이 쓰이는가 (라벨 없는 통제 3종 + 누수 눈금) | 4.3 |
`contrast.py` | 대비별 판별방향 공유도 `cos_b`/`cos_w`, PERM/ZERO 수축 통제 | 4.4 |
`budget.py` | 자기 시행 vs 타인 시행 교환비 곡선 | 4.4 |
`geom.py` | 라플라시안 기저 대조군 `lap`/`pca`/`chan`/`rand` | 4.2 |
`synth.py` | `band_one` / `band_blocks` / concat / `global_one` 표현 비교, `--split loso` | 3.1, 3.5 |
`hier.py` | 계층 LDA, `--split trial/token/time` 분할 통제 | 3.4 |
`hier2.py` | λ(노이즈 공유) × γ(클래스 공유) 표면 | 4.4 |
`onemodel.py` | 통합 객체 4종 비교 (`one_K0`/`one_KT`/`blocks`/concat), K_time 효과 | 4.1 |

리저버 arm은 제거했다 — 두피 라플라시안 W가 랜덤 W를 +0.024로 이겼지만(15/15, p<1e-4) 지연
임베딩에 보태지 못했고(−0.019, p<1e-4), 쓰지 않으므로 코드를 남기지 않았다. 측정 기록은
`onemodel.py` docstring과 `SUMMARY.md` 4.3에 있다.

이 스크립트들이 만든 결론 중 `model/`에 반영된 것: `final_model.py`는 통제를 통과한 부품만
남긴 결과이고, 통과하지 못한 것(피험자 억제, 기하 매끄러움 prior, 라플라시안 기저, 리저버,
비선형 지연 항, γ의 공유 해석)은 전부 빠졌다.
