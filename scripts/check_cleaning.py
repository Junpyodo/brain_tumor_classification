"""정제 코드 시험: 합성 이미지를 일부러 망가뜨린 뒤 cleaning.py가 제대로 고치는지 확인.
저장소 루트에서 실행:  python -m scripts.check_cleaning
(데이터 다운로드 없이 몇 초 안에 끝남)"""
import random

import numpy as np
from PIL import Image

from src.cleaning import clean_splits, find_duplicates
from src.config import LABEL_NAMES


# ---------------------------------------------------------------- 테스트용 오염 생성
def make_dirty(records, label_names=LABEL_NAMES, seed=0, frac=0.03):
    """깨끗한 데이터를 일부러 망가뜨려 정제 코드를 시험한다.
    모드 변경, 이상한 크기, 문자열 라벨, None 이미지, 초소형 이미지, 잘못된 라벨, 중복."""
    rng = random.Random(seed)
    n = len(records)
    k = max(1, int(n * frac))
    ops = ["mode", "resize", "label_str", "none_image", "tiny", "bad_label"]
    assert len(ops) * k <= n, "frac가 너무 큼"
    idx = list(range(n))
    rng.shuffle(idx)
    dirty = [dict(r) for r in records]
    log = {}
    for j, op in enumerate(ops):
        chosen = idx[j * k:(j + 1) * k]
        log[op] = len(chosen)
        for i in chosen:
            r = dirty[i]
            if op == "mode":
                r["image"] = r["image"].convert(rng.choice(["L", "RGBA", "P", "CMYK"]))
            elif op == "resize":
                r["image"] = r["image"].resize((rng.randint(100, 600), rng.randint(100, 600)))
            elif op == "label_str":
                name = label_names[r["label"]]
                variants = [name.upper(), f" {name} ", name.replace("-", "_"), name.capitalize()]
                if name == "no-tumor":
                    variants.append("notumor")
                r["label"] = rng.choice(variants)
            elif op == "none_image":
                r["image"] = None
            elif op == "tiny":
                r["image"] = r["image"].resize((8, 8))
            elif op == "bad_label":
                r["label"] = rng.choice(["unknown", 99, None])
    dup_src = rng.sample(range(n), k)
    dirty.extend(dict(records[i]) for i in dup_src)
    log["duplicates"] = len(dup_src)
    return dirty, log


def inject_cross_split_duplicates(splits, src="train", dst="test", n=10, seed=0):
    """src의 이미지를 dst에 몰래 복사해 넣는다 (누수 시나리오 시험용)."""
    rng = random.Random(seed)
    out = {k: list(v) for k, v in splits.items()}
    out[dst].extend(dict(splits[src][i]) for i in rng.sample(range(len(splits[src])), n))
    return out


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