"""데이터 정제: 형식 통일, 라벨 정리, 중복 이미지 검사/제거.

이 파일은 tensorflow 없이도 동작한다 (PIL, numpy만 사용).
레코드 형식: {"image": PIL.Image, "label": int 또는 str}
splits 형식 : {"train": [레코드...], "validation": [...], "test": [...]}
"""
import hashlib
from collections import defaultdict
from itertools import combinations

import numpy as np
from PIL import Image

from .config import LABEL_ALIASES, LABEL_NAMES, normalize_name


# ---------------------------------------------------------------- 형식 통일
def standardize_image(img, size=None, min_side=32):
    """어떤 모드(L, RGBA, P, CMYK, 16bit 등)의 이미지든 RGB uint8로 통일한다.
    깨졌거나 None이거나 너무 작으면 ValueError."""
    if img is None:
        raise ValueError("image is None")
    try:
        img.load()   # 잘린/손상 파일은 여기서 걸림
    except Exception as e:
        raise ValueError(f"unreadable image: {e}")
    if min(img.size) < min_side:
        raise ValueError(f"image too small: {img.size}")

    if img.mode in ("I;16", "I", "F"):   # 16bit/실수 이미지는 0~255로 스케일
        arr = np.asarray(img).astype(np.float64)
        lo, hi = arr.min(), arr.max()
        arr = (arr - lo) / (hi - lo) * 255 if hi > lo else np.zeros_like(arr)
        img = Image.fromarray(arr.astype(np.uint8), mode="L")
    img = img.convert("RGB")
    if size is not None and img.size != tuple(size):
        img = img.resize(size)
    return img


def normalize_label(label, label_names=LABEL_NAMES):
    """int 또는 문자열 라벨을 표준 int 인덱스로. 해석 불가하면 ValueError."""
    if isinstance(label, (int, np.integer)) and not isinstance(label, bool):
        if 0 <= int(label) < len(label_names):
            return int(label)
        raise ValueError(f"label index out of range: {label}")
    if isinstance(label, str):
        name = normalize_name(label)
        name = LABEL_ALIASES.get(name, name)
        std = [normalize_name(n) for n in label_names]
        if name in std:
            return std.index(name)
    raise ValueError(f"unknown label: {label!r}")


def clean_records(records, label_names=LABEL_NAMES, size=None, min_side=32):
    """레코드 리스트를 정리해 (깨끗한 리스트, 리포트)를 돌려준다."""
    clean = []
    rep = defaultdict(int)
    for i, r in enumerate(records):
        try:
            img = standardize_image(r.get("image"), size=size, min_side=min_side)
        except ValueError:
            rep["dropped_bad_image"] += 1
            continue
        try:
            lab = normalize_label(r.get("label"), label_names)
        except ValueError:
            rep["dropped_bad_label"] += 1
            continue
        if r["image"].mode != "RGB":
            rep["mode_converted"] += 1
        if isinstance(r["label"], str):
            rep["label_str_to_int"] += 1
        clean.append({"image": img, "label": lab, "source_idx": i})
    rep["kept"] = len(clean)
    rep["input"] = len(records)
    return clean, dict(rep)


# ---------------------------------------------------------------- 중복 검사
def image_hash(img, method="dhash", hash_size=16):
    """이미지 지문.
    dhash: 리사이즈/재압축에도 잘 안 변함 (기본). md5: 거의 바이트 단위로 같은 것만."""
    gray = img.convert("L")
    if method == "md5":
        small = gray.resize((32, 32))
        return hashlib.md5(np.asarray(small).tobytes()).hexdigest()
    small = np.asarray(gray.resize((hash_size + 1, hash_size)), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return np.packbits(bits).tobytes().hex()


def find_duplicates(splits, method="dhash"):
    """split 내부 중복, split 간 겹침, 같은 이미지인데 라벨이 다른 경우를 센다."""
    seen = defaultdict(list)    # hash -> [(split, idx)]
    for name, recs in splits.items():
        for i, r in enumerate(recs):
            seen[image_hash(r["image"], method)].append((name, i))

    within = defaultdict(int)
    cross = defaultdict(int)
    conflicts = []
    for locs in seen.values():
        if len(locs) < 2:
            continue
        names = [n for n, _ in locs]
        for n in set(names):
            if names.count(n) > 1:
                within[n] += names.count(n) - 1
        for a, b in combinations(sorted(set(names)), 2):
            cross[f"{a}&{b}"] += 1
        labels = {splits[n][i]["label"] for n, i in locs}
        if len(labels) > 1:
            conflicts.append(locs)
    return {"within": dict(within), "cross": dict(cross), "label_conflicts": conflicts}


def remove_duplicates(splits, priority=("test", "validation", "train"), method="dhash"):
    """중복을 제거한다. priority가 앞선 split의 데이터를 남기고 뒤쪽에서 지운다.
    (test를 건드리지 않고 train/validation에서 지우는 것이 기본)"""
    order = [n for n in priority if n in splits] + [n for n in splits if n not in priority]
    seen, kept, removed = set(), {}, {}
    for name in order:
        keep, rem = [], 0
        for r in splits[name]:
            h = image_hash(r["image"], method)
            if h in seen:
                rem += 1
            else:
                seen.add(h)
                keep.append(r)
        kept[name], removed[name] = keep, rem
    return {n: kept[n] for n in splits}, removed


def clean_splits(splits, label_names=LABEL_NAMES, size=None, method="dhash"):
    """형식/라벨 정리 -> 중복 제거를 한 번에. (정리된 splits, 리포트)"""
    cleaned, report = {}, {"format": {}}
    for name, recs in splits.items():
        cleaned[name], report["format"][name] = clean_records(recs, label_names, size)
    cleaned, removed = remove_duplicates(cleaned, method=method)
    report["duplicates_removed"] = removed
    report["after"] = {n: len(r) for n, r in cleaned.items()}
    return cleaned, report