"""정제 코드 시험: 합성 이미지를 일부러 망가뜨린 뒤 cleaning.py가 제대로 고치는지 확인.
저장소 루트에서 실행:  python -m scripts.check_cleaning
(데이터 다운로드 없이 몇 초 안에 끝남)"""
import numpy as np
from PIL import Image

from src.cleaning import (clean_splits, find_duplicates, inject_cross_split_duplicates,
                          make_dirty)
from src.config import LABEL_NAMES

rs = np.random.RandomState(0)


def fake_split(n):
    return [{"image": Image.fromarray(rs.randint(0, 256, (64, 64, 3), dtype=np.uint8)),
             "label": i % len(LABEL_NAMES)} for i in range(n)]


splits = {"train": fake_split(300), "validation": fake_split(100), "test": fake_split(100)}

# 1) train 이미지 10장을 test에 몰래 복사 (데이터 누수 시나리오)
splits = inject_cross_split_duplicates(splits, "train", "test", n=10)
# 2) validation만 일부러 지저분하게 만들기
splits["validation"], log = make_dirty(splits["validation"], seed=1, frac=0.05)
print("오염 로그:", log)

print("\n[정제 전 중복 검사]")
print({k: v for k, v in find_duplicates(
    {n: [r for r in recs if r["image"] is not None] for n, recs in splits.items()}).items()
    if k != "label_conflicts"})

cleaned, report = clean_splits(splits, size=(64, 64))
print("\n[정제 리포트]")
for k, v in report.items():
    print(k, v)

after = find_duplicates(cleaned)
print("\n[정제 후 중복 검사]", {k: v for k, v in after.items() if k != "label_conflicts"})

# 자동 확인
fmt = report["format"]["validation"]
assert fmt["dropped_bad_image"] == log["none_image"] + log["tiny"], fmt
assert fmt["dropped_bad_label"] == log["bad_label"], fmt
assert fmt["label_str_to_int"] == log["label_str"], fmt
assert not after["cross"] and not after["within"], after
assert all(r["image"].mode == "RGB" and r["image"].size == (64, 64)
           for recs in cleaned.values() for r in recs)
assert all(isinstance(r["label"], int) for recs in cleaned.values() for r in recs)
print("\n모든 검사 통과")
