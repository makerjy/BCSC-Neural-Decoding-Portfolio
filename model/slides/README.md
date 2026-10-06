# slides/ — 발표자료와 포스터 그림

```bash
python slides/make_figs.py --compute     # 실데이터가 필요한 값 계산 (한 번만, 몇 분)
python slides/make_figs.py               # 그림 15장 렌더
python slides/build_deck.py              # neural_speech_decoding.pptx (19장)
python slides/qa_deck.py                 # 여백 이탈 · 도형 겹침 · 텍스트 넘침 검사
python slides/preview.py                 # 근사 렌더링 (눈으로 확인용)
```

## 파일

| | |
|---|---|
`make_figs.py` | 그림 15장. `--compute`가 `npz2`에서 실제 값을 계산해 `cache.npz`에 저장하고, 나머지는 `SUMMARY.md`·`logs/`의 숫자를 읽는다 |
`build_deck.py` | 덱 생성. 레이아웃 수치가 전부 여기 있다 |
`qa_deck.py` | 기하 검사. **이 환경에는 LibreOffice가 없어 렌더링 QA가 불가능하므로** 그 대역이다 |
`preview.py` | `.pptx`를 되읽어 matplotlib으로 근사 렌더. 한글은 네모로 나오지만(폰트 없음) 배치는 실제와 같다 |
`figs/` | PNG 15장 (200 dpi) |
`cache.npz` | 실데이터 파생값. `--compute`로 재생성 |

## 스타일

**서체** — matplotlib이 Computer Modern(`cmr10`)을 포함하므로 그림은 진짜 LaTeX 서체로 나온다.
슬라이드 텍스트는 라틴 Times New Roman + 한글 바탕(`<a:ea>`). cmr10에 없는 글리프가 있다:
`·`는 `$\cdot$`로, 밑줄과 `~`는 쓰지 말 것.

**색** — 딱 하나의 뜻으로만 쓴다. 강조색(`#A62B1F`)은 **"우리 것"**, 회색은 베이스라인과 대조군.
두 번째 표현에만 파랑(`#20548F`). 색 막대나 제목 밑줄은 넣지 않는다.

## 그림 15장의 출처 — 세 등급

포스터에서 "이거 실제 데이터인가요"라는 질문이 반드시 나온다. 등급이 다르니 구분해서 답할 것.

### ① 실데이터에서 직접 계산 (`--compute` 필요, `npz2` → `cache.npz`)

| 그림 | 무엇 |
|---|---|
`spd` | 실제 36×36 객체 + **항목별 F = 클래스간/클래스내**. φ 교차 블록이 공간 블록의 약 2배 (2.99·2.76 대 1.59·1.35) — `SUMMARY.md` 4.1 |
`patterns` | 공간 필터 상위 6성분의 두피 지도. **필터가 아니라 패턴**으로 변환해 그린다 (Haufe et al. 2014) — 필터를 그대로 그리면 해석이 틀린다 |
`confusion` | 11×11 혼동행렬 (fold 0, 피험자별 LDA 합산). 모음/자음 블록과 s↔z · t↔d · p↔b 혼동 |
`montage` | 실제 전극 좌표 + **실제 단일시행 3개와 그 평균** |

### ② 로그를 파싱해서 그린 것 (측정값이지만 이 세션에서 재계산하진 않음)

`paired` · `phi_ablation` · `ktime` · `splits` · `budget` · `results` · `emergence` ·
`reservoir_sweep`

**전부 `logs/`에서 정규식으로 읽는다. 리터럴로 타이핑된 숫자는 하나도 없다.**
`python slides/make_figs.py --verify`가 파싱된 값을 전부 출력하므로 로그와 대조할 수 있다.

> 왜 이렇게 바꿨나: 처음에는 손으로 옮겨 적었고 **`paired`의 15개 폴드 중 10개가 틀렸다**
> (평균 0.295로 그려짐, 실제 0.305). `reservoir_sweep`은 행마다 leak 값이 뒤섞여 있었다.
> 같은 사고를 구조적으로 막으려고 파싱으로 전환했다.

### ③ 순수 도식 — 데이터 없음

`arch` · `bands` — 상자와 화살표다. 안에 적힌 숫자(0.270 / 0.289 / 0.306, 차원 2495 / 2911 /
5406)는 측정값이지만 **그림 자체는 그린 것**이다.

`phi` — 중간이다. DCT 기저를 `sdcore.phi_rows`와 **같은 공식으로 계산**한 것이라 모델이 실제로
쓰는 파형이지만, 측정값은 아니다.

### 주의 — `results`의 두 프로토콜

`designed` 0.245와 `best` 0.306은 **미접촉 최종 test셋**, EEGNet 0.225는 **15폴드 CV**다.
`holdout.py`의 `MODELS`에 EEGNet이 들어간 것이 로그보다 나중이라 EEGNet의 미접촉 점수가 아직
없다. 그림에 빗금과 라벨로 구분해 뒀다. **셋을 같은 프로토콜로 말하지 말 것.**

## 포스터로 옮길 때

덱은 16:9지만 그림은 전부 벡터로 다시 뽑을 수 있다 — `save()`에서
`fig.savefig(p.replace(".png", ".svg"))`를 추가하면 A0 인쇄용 SVG가 나온다.
`svg.fonttype`을 `"none"`으로 두면 Figma/Inkscape에서 텍스트를 그대로 편집할 수 있다.

## 알려진 제약

- **렌더링 QA를 못 했다.** `soffice`가 없어 실제 PowerPoint 출력을 본 적이 없다.
  `qa_deck.py`가 여백·겹침·넘침을 기계적으로 확인했고 `preview.py`로 배치를 눈으로 봤지만,
  줄바꿈은 PowerPoint가 다시 하므로 **한 번 열어서 확인할 것.**
- 한글 서체가 바탕으로 지정돼 있다. Mac에서는 대체 서체가 잡힐 수 있다.
