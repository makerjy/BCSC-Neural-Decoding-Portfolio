import argparse
from collections import Counter
from pathlib import Path

import numpy as np


def flatten_nested(value):
    """중첩된 numpy/list/object 값을 문자열 목록으로 펼친다."""
    result = []

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return flatten_nested(value.item())

        for item in value:
            result.extend(flatten_nested(item))

    elif isinstance(value, (list, tuple)):
        for item in value:
            result.extend(flatten_nested(item))

    elif value is not None:
        result.append(str(value))

    return result


def print_unique(data, key, filename_only=False):
    if key not in data.files:
        print(f"\n[{key}] key 없음")
        return

    values = flatten_nested(data[key])

    if filename_only:
        values = [Path(v).name for v in values]

    counts = Counter(values)

    print("\n" + "=" * 70)
    print(f"{key} 종류: {len(counts)}개")
    print("=" * 70)

    for value, count in counts.most_common():
        print(f"{count:8d}  {value}")


def main():
    parser = argparse.ArgumentParser(
        description="NPZ 파일의 task/event/condition 구조 확인"
    )
    parser.add_argument("npz_path", help="확인할 .npz 파일 경로")
    parser.add_argument(
        "--examples",
        type=int,
        default=10,
        help="trial별 event 예시 출력 개수 (기본값: 10)",
    )
    args = parser.parse_args()

    npz_path = Path(args.npz_path).expanduser().resolve()

    if not npz_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없음: {npz_path}")

    print(f"\n파일: {npz_path}")
    print(f"크기: {npz_path.stat().st_size / 1024 / 1024:.2f} MB")

    with np.load(npz_path, allow_pickle=True) as data:
        print("\n" + "=" * 70)
        print("1. NPZ 전체 구조")
        print("=" * 70)

        for key in data.files:
            arr = data[key]
            print(
                f"{key:32s} "
                f"shape={str(arr.shape):20s} "
                f"dtype={arr.dtype}"
            )

        # trial별로 저장되었을 가능성이 높은 정보
        print_unique(data, "subject_per_trial")
        print_unique(data, "condition_per_trial")
        print_unique(data, "y_condition")
        print_unique(data, "source_set_per_trial", filename_only=True)

        # task 정보가 들어 있을 가능성이 높은 event 정보
        print_unique(data, "event_code")
        print_unique(data, "event_label")

        # 명시적인 task 관련 key가 있는지 탐색
        print("\n" + "=" * 70)
        print("2. task 관련 key 검색")
        print("=" * 70)

        keywords = [
            "task",
            "trial_type",
            "stimulus",
            "word",
            "pseudo",
            "phoneme",
            "syllable",
            "label",
        ]

        task_keys = [
            key
            for key in data.files
            if any(keyword in key.lower() for keyword in keywords)
        ]

        if task_keys:
            for key in task_keys:
                print(f"- {key}")
        else:
            print("명시적인 task 관련 key를 찾지 못함")

        # event code-label 대응 확인
        if "event_code" in data.files and "event_label" in data.files:
            codes = data["event_code"]
            labels = data["event_label"]

            print("\n" + "=" * 70)
            print("3. trial별 event_code / event_label 예시")
            print("=" * 70)

            n_examples = min(args.examples, len(codes), len(labels))

            for i in range(n_examples):
                print(f"\nTrial {i}")
                print(f"  event_code : {codes[i]}")
                print(f"  event_label: {labels[i]}")

            pair_counts = Counter()

            if len(codes) == len(labels):
                for trial_codes, trial_labels in zip(codes, labels):
                    flat_codes = flatten_nested(trial_codes)
                    flat_labels = flatten_nested(trial_labels)

                    if len(flat_codes) == len(flat_labels):
                        pair_counts.update(zip(flat_codes, flat_labels))

            if pair_counts:
                print("\n" + "=" * 70)
                print("4. event_code ↔ event_label 대응")
                print("=" * 70)

                for (code, label), count in pair_counts.most_common():
                    print(f"{count:8d}  {code}  ->  {label}")
            else:
                print(
                    "\nevent_code와 event_label 구조가 달라서 "
                    "자동 대응표는 만들지 못함"
                )

        print("\n" + "=" * 70)
        print("5. task 후보 단어 검색")
        print("=" * 70)

        candidate_words = [
            "phoneme",
            "phonemes",
            "word",
            "words",
            "pseudo",
            "pseudoword",
            "pseudowords",
            "triplet",
            "triplets",
            "syllable",
            "cv",
            "vowel",
            "consonant",
        ]

        found = Counter()

        for key in [
            "event_code",
            "event_label",
            "source_set_per_trial",
            "condition_per_trial",
        ]:
            if key not in data.files:
                continue

            for value in flatten_nested(data[key]):
                lower_value = value.lower()

                for word in candidate_words:
                    if word in lower_value:
                        found[(key, word, value)] += 1

        if found:
            for (key, word, original), count in found.most_common():
                print(
                    f"{count:8d}  key={key:22s} "
                    f"후보={word:12s} 값={original}"
                )
        else:
            print("현재 NPZ에서 task를 나타내는 후보 단어를 찾지 못함")
            print(
                "이 경우 task 정보가 원본 Tab.csv에만 있거나 "
                "다른 메타데이터 필드에 있을 수 있습니다."
            )


if __name__ == "__main__":
    main()
